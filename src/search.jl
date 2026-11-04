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
    Dict(k => copy(v) for (k, v) in st.reads), st.closed_through, copy(st.hidden), st.infer, nothing, nothing, copy(st.reopened))
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

"""
Depth-first search settings: alternatives need probability >= eps; at most `budget` failures per segment;
`jump`: conflict-directed backjumping (back to the latest choice point among the failure's causes);
`order`: a choice point's values in probability order (:prob) or by the product score (:product: an
attachment's heads by P(head) times its uncommitted head properties' probabilities of the values the head
implies; a head property's values by P(value) times the attachment probability of the heads consistent with it).
"""
struct DFS
  eps::Float64
  budget::Int
  jump::Bool
  order::Symbol
  epshead::Float64                                 # the threshold for head-property decisions (eps if NaN)
end
DFS(; eps = 0.05, budget = 200, jump = false, order = :prob, epshead = NaN) = DFS(eps, budget, jump, order, epshead)

const DFS_STATS = Dict(:segments => 0, :choicepoints => 0, :failures => 0, :backtracked => 0, :gaveup => 0,
  :jumps => 0, :jumped => 0, :reordered => 0)

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

"Is this attachment (or, with forward checking, head-property) decision a failure: ready, no candidate, and no other head can still appear?"
function attach_failed(st::State, key::AgendaKey)
  d = decisiondef(st, key)
  (d.name == :attach || (FWDCHECK[] && haskey(ALLHEADSETS, d.name))) || return false
  t = key[3]
  d.ready(st, t) && isempty(d.candidates(st, t)) && !more_heads_possible(st, t)
end

"activate!, but the failed key (and nothing done) if a new attachment decision has already failed; else nothing."
function activate_strict!(st::State)
  k = findfirst(key -> attach_failed(st, key), st.fresh)
  k === nothing || return st.fresh[k]
  activate!(st)
  nothing
end

"rescore!, but the failed key as soon as a rescored attachment decision has failed; else nothing."
function rescore_strict!(st::State, model)
  for key in collect(keys(st.agenda))
    reads = get(st.reads, key, nothing)
    if reads !== nothing && haskey(st.best, key) && !any(cell -> read_changed(st, cell), reads)
      continue
    end
    choice, margin = tracked_decide(model, st, key)
    margin == -Inf && attach_failed(st, key) && return key
    st.agenda[key] = margin
    st.best[key] = choice
  end
  nothing
end

"""
Failure diagnosis (DFS_DIAG[] = true): why token t's attachment has no candidate.  Every possible head
(same sentence, plus the root if the sentence has none yet) is put down to the first filter that excludes
it: :distance, :cycle, :sentence_end (an E gap between), a committed head property (:head_dir etc.), or
:projective (crosses a committed arc).  Also: t's committed head properties, and for each, how many choice
points back it was chosen (nothing: not a choice point, e.g. forced or inferred).
"""
function diagnose(st::State, key::AgendaKey, stack)
  toks = st.layers[:token]
  t = key[3]
  s = toks[:sentence, t]
  hasroot = s != PENDING && any(u -> u != t && toks[:sentence, u] == s && toks[:attach, u] == NONE, 1:length(toks))
  heads = UInt32[u for u in 1:length(toks) if u != t && toks[:sentence, u] == s]
  hasroot || pushfirst!(heads, NONE)
  ft = Int(toks[:first_char, t])
  why = Dict{Symbol,Int}()
  names = headnames(toks)
  for h in heads
    r = if h != NONE && abs(Int(toks[:first_char, h]) - ft) > MAX_LINK_CHARS
      :distance
    elseif h != NONE && creates_cycle(st, t, Int(h))
      :cycle
    elseif h != NONE && crosses_E(st, t, Int(h))
      :sentence_end
    else
      bad = findfirst(n -> (v = toks[n, t]; v != PENDING && !head_consistent_one(st, t, h, n)), names)
      bad !== nothing ? names[bad] : crosses_arc(st, t, h) ? :projective : :none
    end
    why[r] = get(why, r, 0) + 1
  end
  props = Dict(n => toks[n, t] for n in names if toks[n, t] != PENDING)
  back = Dict{Symbol,Any}()
  for (n, v) in props
    k = findfirst(d -> d.name == n, toks.spec.decisions)
    j = findlast(cp -> cp.key == (:token, k, t), stack)
    back[n] = j === nothing ? nothing : length(stack) - j
  end
  (token = tokstr(st, t), nheads = length(heads), why = why, props = Dict(n => ALLHEADSETS[n][v] for (n, v) in props),
    back = back, depth = length(stack))
end

"""
The decisions whose commits excluded token t's possible heads (the conflict set of t's failed attachment):
its committed head properties; for head-tag conflicts, the candidate head's tag; for cycles, the
attachments along the path from the head back to t; for crossing arcs, the crossing tokens' attachments;
the token attached to the root (if that excluded the root); the E gaps between t and a head.
"""
function conflict_keys(st::State, key::AgendaKey)
  toks = st.layers[:token]
  decs = toks.spec.decisions
  kof = Dict(d.name => k for (k, d) in enumerate(decs))
  t = key[3]
  s = toks[:sentence, t]
  out = Set{AgendaKey}()
  root = findfirst(u -> u != t && toks[:sentence, u] == s && toks[:attach, u] == NONE, 1:length(toks))
  s != PENDING && root !== nothing && push!(out, (:token, 1, root))
  names = headnames(toks)
  pos(u) = Int(toks[:first_char, u])
  for h in (u for u in 1:length(toks) if u != t && toks[:sentence, u] == s)
    abs(pos(h) - pos(t)) > MAX_LINK_CHARS && continue
    if creates_cycle(st, t, h)
      u = h
      while u != t                                     # the attachments on the path from h up to t
        push!(out, (:token, 1, u))
        v = toks[:attach, u]
        (v == NONE || v == PENDING) && break
        u = Int(v)
      end
      continue
    end
    if crosses_E(st, t, h)
      chars = st.layers[:char]
      l, r = pos(t) < pos(h) ? (t, h) : (h, t)
      for j in Int(toks[:last_char, l]):Int(toks[:first_char, r])-1
        chars[:gap, j] == GAP_E && push!(out, (:char, 1, j))
      end
      continue
    end
    bad = findfirst(n -> toks[n, t] != PENDING && !head_consistent_one(st, t, UInt32(h), n), names)
    if bad !== nothing
      n = names[bad]
      push!(out, (:token, kof[n], t))
      n in (:head_upos, :head_xpos) && push!(out, (:token, kof[n == :head_upos ? :upos : :xpos], h))
      continue
    end
    a, b = minmax(pos(t), pos(h))                      # crossing arcs
    for u in 1:length(toks)
      u == t && continue
      g = toks[:attach, u]
      g == PENDING && continue
      c, d = g == NONE ? (0, pos(u)) : minmax(pos(u), pos(Int(g)))
      (a < c < b < d || c < a < d < b) && push!(out, (:token, 1, u))
    end
  end
  for n in names                                       # (and t's committed head properties in any case)
    toks[n, t] != PENDING && push!(out, (:token, kof[n], t))
  end
  out
end

"""
A choice point's values ordered by the product score (DFS order :product), or nothing to keep the
probability order: attachments and head properties only.
"""
function product_order(model, st::State, key::AgendaKey, cands, lps, eps)
  toks = st.layers[:token]
  decs = toks.spec.decisions
  t = key[3]
  name = decs[key[2]].name
  keep = [i for i in eachindex(cands) if exp(lps[i]) >= eps]
  if name == :attach
    sc = Dict{UInt32,Float64}(cands[i] => lps[i] for i in keep)
    for (k, d) in enumerate(decs)
      haskey(ALLHEADSETS, d.name) && toks[d.name, t] == PENDING || continue
      dist = commit_dist(model, st, (:token, k, t))
      dist === nothing && continue
      vals, vlps = dist
      for c in keys(sc)
        v = head_value(st, t, c, d.name)
        v == PENDING && continue
        j = findfirst(==(v), vals)
        sc[c] += j === nothing ? log(1e-12) : vlps[j]
      end
    end
    return sort(collect(keys(sc)); by = c -> -sc[c])
  elseif haskey(ALLHEADSETS, name) && toks[:attach, t] == PENDING
    adist = commit_dist(model, st, (:token, 1, t))
    adist === nothing && return nothing
    hs, hlps = adist
    sc = Dict{UInt32,Float64}()
    for i in keep
      v = cands[i]
      sup = sum((exp(hlps[j]) for j in eachindex(hs) if (hv = head_value(st, t, hs[j], name)) == v || hv == PENDING); init = 0.0)
      sc[v] = lps[i] + log(sup + 1e-12)
    end
    return sort(collect(keys(sc)); by = c -> -sc[c])
  end
  nothing
end

const DFS_DIAG = Ref(false)
const DFS_LOG = Any[]

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
      fk = activate_strict!(st)
      fk === nothing && isempty(st.agenda) && break
      fk === nothing && (fk = rescore_strict!(st, model))
      if fk === nothing
        key, margin = first(st.agenda)
        margin == -Inf && break
        choice = st.best[key]
        if margin < Inf
          dist = commit_dist(model, st, key)
          if dist !== nothing
            cands, lps = dist
            ek = !isnan(opts.epshead) && haskey(ALLHEADSETS, decisiondef(st, key).name) ? opts.epshead : opts.eps
            others = [(c, l) for (c, l) in zip(cands, lps) if c != choice && exp(l) >= ek]
            alts = [choice; first.(sort!(others; by = x -> -x[2]))]
            if opts.order == :product && length(alts) > 1
              po = product_order(model, st, key, cands, lps, opts.eps)
              if po !== nothing && !isempty(po)
                po[1] != choice && (DFS_STATS[:reordered] += 1)
                alts = po
                choice = po[1]
              end
            end
            if length(alts) > 1
              push!(stack, ChoicePoint(length(TRAIL[]), snapshot(st), key, alts, 1))
              DFS_STATS[:choicepoints] += 1
            end
          end
        end
        commit_choice!(st, key, choice)
        continue
      end
      # a failure: back up to the latest choice point with a value left
      DFS_DIAG[] && length(DFS_LOG) < 100_000 && push!(DFS_LOG, diagnose(st, fk, stack))
      failures += 1
      DFS_STATS[:failures] += 1
      if opts.jump && !isempty(stack)                 # backjump: to the latest choice point among the failure's causes
        ck = conflict_keys(st, fk)
        j = findlast(cp -> cp.key in ck, stack)
        if j !== nothing && j < length(stack)
          DFS_STATS[:jumps] += 1
          DFS_STATS[:jumped] += length(stack) - j
          resize!(stack, j)
        end
      end
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
