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
