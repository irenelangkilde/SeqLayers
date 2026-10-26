# search.jl  --  limited-discrepancy search over the easy-first decoder
#
# The greedy decoder (solve!) commits the most confident decision, propagates, and repeats; a wrong
# early commit can't be undone.  LDS keeps a copy of the state just before the `k` least confident
# commits of the greedy run (the branch points), then for each reruns the decoder from that copy
# with the decision's runner-up value committed instead.  Every path is scored by the log
# probabilities of its commits (choose's softmax over the candidates, plus wait for open
# decisions), summed or averaged per commit; the best-scoring path's final state wins.  One
# discrepancy per path: the branches don't branch again.
#
# States are copied rather than trailed: a segment's state is a few small grids plus the agenda,
# and constraints may add columns, which an undo trail would have to reverse as well.

"""
Search settings: retry the `k` least confident commits (margin below `maxmargin`; token-layer
decisions only if `tokenonly`) with their runner-up; paths compared by the sum (`score = :sum`) or
the mean (`:mean`) of their commits' log probabilities.
"""
struct LDS
  k::Int
  maxmargin::Float64
  tokenonly::Bool
  score::Symbol
end
LDS(k; maxmargin = Inf, tokenonly = true, score = :sum) = LDS(k, maxmargin, tokenonly, score)

"Counts over a run: segments searched, branches tried, segments where a branch won."
const LDS_STATS = Dict(:segments => 0, :branches => 0, :switched => 0)

"A copy of a decoder state that later commits can't touch (doc, lexicon and specs are shared)."
function copy_state(st::State)
  layers = Dict{Symbol,Sequence}(n => typeof(s)(s.spec, copy(s.grid), s.ncols) for (n, s) in st.layers)
  State(st.doc, st.span, st.final, st.edge, layers, st.order, st.lex, copy(st.agenda), copy(st.best), copy(st.fresh),
    Dict(k => copy(v) for (k, v) in st.reads), st.closed_through, copy(st.hidden), st.infer, nothing, nothing)
end

"Make `st` the state `src` (in place: the caller holds `st`)."
function adopt!(st::State, src::State)
  for f in (:layers, :agenda, :best, :fresh, :reads, :closed_through, :hidden, :memo, :relcache)
    setfield!(st, f, getfield(src, f))
  end
  st
end

"Decision `key`'s candidates and their log probabilities as `choose` sees them (wait not listed)."
function commit_dist(model, st::State, key::AgendaKey)
  got = Ref{Any}(nothing)
  CHOOSE_HOOK[] = (cands, scores) -> (got[] = (cands, scores))
  try
    decide(model, st, key)
  finally
    CHOOSE_HOOK[] = nothing
  end
  got[] === nothing && return nothing
  cands, scores = got[]
  mx = maximum(scores)
  lz = mx + log(sum(s -> exp(s - mx), scores))
  (cands, [scores[j] - lz for j in eachindex(cands)])
end

mutable struct Branch
  margin::Float64
  key::AgendaKey
  snap::State            # just before the commit
  alt::UInt32            # the runner-up
  altlp::Float64
  prefix::Float64        # summed log probability of the commits before
  nprefix::Int
end

"""
The greedy decoder, also summing its commits' log probabilities (returned with their number).
With `record`, keeps the `lds.k` least confident branch points in it.
"""
function greedy!(st::State, model; record = nothing, lds = nothing)
  total, n = 0.0, 0
  while true
    activate!(st)
    isempty(st.agenda) && break
    rescore!(st, model)
    key, margin = first(st.agenda)
    margin == -Inf && break
    choice = st.best[key]
    lp, alt, altlp = 0.0, nothing, -Inf
    if margin < Inf
      dist = commit_dist(model, st, key)
      if dist !== nothing
        cands, lps = dist
        j = findfirst(==(choice), cands)
        j === nothing || (lp = lps[j])
        for (c, l) in zip(cands, lps)
          c != choice && l > altlp && ((alt, altlp) = (c, l))
        end
      end
    end
    if record !== nothing && lds.k > 0 && alt !== nothing && margin < lds.maxmargin && (!lds.tokenonly || key[1] == :token) &&
       (length(record) < lds.k || margin < maximum(b.margin for b in record))
      push!(record, Branch(margin, key, copy_state(st), alt, altlp, total, n))
      length(record) > lds.k && deleteat!(record, argmax([b.margin for b in record]))
    end
    delete!(st.agenda, key); delete!(st.best, key); delete!(st.reads, key)
    commit!(st, key, choice)
    total += lp; n += 1
  end
  total, n
end

pathscore(lds::LDS, total, n) = lds.score == :mean ? total / max(n, 1) : total

"Solve `st` with limited-discrepancy search (see the top of this file)."
function solve_lds!(st::State, model, lds::LDS)
  record = Branch[]
  total, n = greedy!(st, model; record, lds)
  best, bestst = pathscore(lds, total, n), nothing
  LDS_STATS[:segments] += 1
  for b in record
    sb = copy_state(b.snap)
    delete!(sb.agenda, b.key); delete!(sb.best, b.key); delete!(sb.reads, b.key)
    commit!(sb, b.key, b.alt)
    rest, nr = greedy!(sb, model)
    sc = pathscore(lds, b.prefix + b.altlp + rest, b.nprefix + 1 + nr)
    LDS_STATS[:branches] += 1
    sc > best && ((best, bestst) = (sc, sb))
  end
  bestst === nothing || (adopt!(st, bestst); LDS_STATS[:switched] += 1)
  st
end


## ======== depth-first search with failures (DFS)
#
# Every commit with a plausible alternative (probability >= eps besides the chosen value) is a
# choice point: the grids are trailed (TRAIL: every write since the choice point can be undone) and
# the small bookkeeping (agenda, best values, fresh decisions, reads, closed_through) is copied.
# A failure -- an attachment with no candidate left while no other head can appear (which also
# covers committed head properties no head satisfies, cycles, a second root, and with projectivity
# a crossing arc, since candidates exclude those) -- backs up to the latest choice point and tries
# its next value in probability order; a choice point out of values is popped.  The first complete
# parse wins.  After `budget` failures (or with nothing left to try) the segment is finished by the
# plain greedy decoder from the first choice point -- what solve! would have produced.

"Depth-first search settings: alternatives need probability >= eps; at most `budget` failures per segment."
struct DFS
  eps::Float64
  budget::Int
end
DFS(; eps = 0.05, budget = 200) = DFS(eps, budget)

const DFS_STATS = Dict(:segments => 0, :choicepoints => 0, :failures => 0, :backtracked => 0, :gaveup => 0)

mutable struct ChoicePoint
  mark::Int
  snap::Tuple
  key::AgendaKey
  alts::Vector{UInt32}
  i::Int
end

snapshot(st::State) = (copy(st.agenda), copy(st.best), copy(st.fresh), copy(st.reads), st.closed_through)

"Undo every trailed write after `mark` (newest first)."
function undo!(mark::Int)
  t = TRAIL[]
  while length(t) > mark
    (s, r, i, old) = pop!(t)
    r == 0 ? (s.ncols = i) : (s.grid[r, i] = old)
  end
end

"Take st back to choice point cp (before its decision was committed)."
function restore!(st::State, cp::ChoicePoint)
  undo!(cp.mark)
  a, b, f, r, ct = cp.snap
  st.agenda, st.best, st.fresh, st.reads, st.closed_through = copy(a), copy(b), copy(f), copy(r), ct
  st.relcache = nothing
  refresh!(st)
  st
end

commit_choice!(st::State, key::AgendaKey, v::UInt32) =
  (delete!(st.agenda, key); delete!(st.best, key); delete!(st.reads, key); commit!(st, key, v))

"Is this attachment decision a failure: ready, no candidate, and no other head can still appear?"
function attach_failed(st::State, key::AgendaKey)
  d = decisiondef(st, key)
  d.name == :attach || return false
  t = key[3]
  d.ready(st, t) && isempty(d.candidates(st, t)) && !more_heads_possible(st, t)
end

"activate!, but false (and nothing done) if a new attachment decision has already failed."
function activate_strict!(st::State)
  any(key -> attach_failed(st, key), st.fresh) && return false
  activate!(st)
  true
end

"rescore!, but false as soon as a rescored attachment decision has failed."
function rescore_strict!(st::State, model)
  for key in collect(keys(st.agenda))
    reads = get(st.reads, key, nothing)
    if reads !== nothing && haskey(st.best, key) && !any(cell -> read_changed(st, cell), reads)
      continue
    end
    choice, margin = tracked_decide(model, st, key)
    margin == -Inf && attach_failed(st, key) && return false
    st.agenda[key] = margin
    st.best[key] = choice
  end
  true
end

"Solve `st` by depth-first search with failures (see above)."
function solve_dfs!(st::State, model, opts::DFS)
  DFS_STATS[:segments] += 1
  stack = ChoicePoint[]
  failures = 0
  TRAIL[] = Tuple{Any,Int,Int,UInt32}[]
  STRICT[] = true
  giveup = false
  try
    while true
      ok = activate_strict!(st)
      ok && isempty(st.agenda) && break
      ok = ok && rescore_strict!(st, model)
      if ok
        key, margin = first(st.agenda)
        margin == -Inf && break
        choice = st.best[key]
        if margin < Inf
          dist = commit_dist(model, st, key)
          if dist !== nothing
            cands, lps = dist
            others = [(c, l) for (c, l) in zip(cands, lps) if c != choice && exp(l) >= opts.eps]
            if !isempty(others)
              sort!(others; by = x -> -x[2])
              push!(stack, ChoicePoint(length(TRAIL[]), snapshot(st), key, [choice; first.(others)], 1))
              DFS_STATS[:choicepoints] += 1
            end
          end
        end
        commit_choice!(st, key, choice)
        continue
      end
      # a failure: back up to the latest choice point with a value left
      failures += 1
      DFS_STATS[:failures] += 1
      resumed = false
      while failures <= opts.budget && !isempty(stack)
        cp = stack[end]
        restore!(st, cp)
        if cp.i < length(cp.alts)
          cp.i += 1
          DFS_STATS[:backtracked] += 1
          commit_choice!(st, cp.key, cp.alts[cp.i])
          resumed = true
          break
        end
        pop!(stack)
      end
      resumed && continue
      giveup = true
      break
    end
    if giveup                                         # finish greedily, from the first choice point if any
      DFS_STATS[:gaveup] += 1
      if !isempty(stack)
        cp = stack[1]
        restore!(st, cp)
        commit_choice!(st, cp.key, cp.alts[1])
      end
    end
  finally
    TRAIL[] = nothing
    STRICT[] = false
  end
  giveup && solve!(st, model)
  st
end
