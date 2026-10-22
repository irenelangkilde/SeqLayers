# models.jl
#
# A model turns a pending decision into (best candidate, margin).  The margin is what
# the agenda orders by, so it must be comparable across layers.  -Inf means "not ready".

abstract type AbstractModel end

"""
Generic decision.  `score` gives each candidate's log-odds of being the right one; these are
normalized into probabilities over the candidates, and the margin is the log-odds that the
best candidate is right.  An open decision (more candidates may appear) also includes a
"wait" option scored by `waitscore`, standing in for candidates that don't exist yet.
"""
function decide(m::AbstractModel, st::State, key::AgendaKey)
  d, i = decisiondef(st, key), key[3]
  d.ready(st, i) || return (NONE, -Inf)
  cands = d.candidates(st, i)
  isempty(cands) && return (NONE, -Inf)
  choose(cands, [score(m, st, key, c) for c in cands], d.open ? waitscore(m, st, key) : nothing)
end

"Best candidate and the log-odds that it is right, from candidates' log-odds (plus an optional wait option)."
function choose(cands, scores::Vector{Float64}, wait)
  b = argmax(scores)
  wait === nothing || (scores = [scores; wait])
  length(scores) == 1 && return (cands[b], Inf)          # the only possibility
  mx = maximum(scores)
  z = sum(s -> exp(s - mx), scores)
  pb = exp(scores[b] - mx) / z
  (cands[b], pb >= 1 ? Inf : log(pb) - log1p(-pb))
end

"Log-odds of a candidate that can't be seen yet (for open decisions)."
waitscore(::AbstractModel, st::State, key::AgendaKey) = 0.0

"""
Odds smoothing.  `c1`: times the feature described the chosen candidate; `c2`: times it
described a candidate that wasn't chosen; `e`: the expected (prior) odds.  Moves the raw odds
r = c1/c2 toward e by one count, adjusting whichever count is smaller, never past e:

    e                      if r == e
    min(e, c1/(c2 - 1))    if r < e and c1 > c2 and c2 > 1
    min(e, (c1 + 1)/c2)    if r < e and c1 <= c2
    max(e, (c1 - 1)/c2)    if r > e and c1 <= c2
    max(e, c1/(c2 + 1))    if r > e and c1 > c2
    e                      otherwise (including no observations, and r < e with c2 <= 1)
"""
function smoothOdds(c1, c2, e)
  r = c1 / c2                      # NaN when both are 0: no case below matches
  if r == e
    e
  elseif r < e && c1 > c2 && c2 > 1
    min(e, c1 / (c2 - 1))
  elseif r < e && c1 <= c2
    min(e, (c1 + 1) / c2)
  elseif r > e && c1 <= c2
    max(e, (c1 - 1) / c2)
  elseif r > e && c1 > c2
    max(e, c1 / (c2 + 1))
  else
    e
  end
end

"""
Naive Bayes in odds form.  Each feature key counts how often it described the chosen
candidate (`chosen`) vs. a candidate that wasn't chosen (`unchosen`).  Priors count the same
for each decision type, and for labeled decisions per label.  A candidate's log-odds are

    log(prior odds) + Σ_features [log(smoothOdds(chosen, unchosen, prior odds)) - log(prior odds)]

so a feature never seen in training contributes nothing.  Placeholder for the
odds/mutual-odds machinery.
"""
struct CountModel <: AbstractModel
  chosen::Dict{UInt64,Float64}
  unchosen::Dict{UInt64,Float64}
  prior::Dict{UInt64,Tuple{Float64,Float64}}      # prior key -> (chosen, unchosen)
end
CountModel() = CountModel(Dict{UInt64,Float64}(), Dict{UInt64,Float64}(), Dict{UInt64,Tuple{Float64,Float64}}())

typekey(lname::Symbol, k::Int) = hash((String(lname), k))
labelkey(lname::Symbol, k::Int, c::UInt32) = hash((String(lname), k, c))
priorkey(d::DecisionDef, lname::Symbol, k::Int, c::UInt32) = d.labeled ? labelkey(lname, k, c) : typekey(lname, k)

function priorodds(m::CountModel, pk::UInt64)
  (c, oc) = get(m.prior, pk, (0.0, 0.0))
  (c + 0.5) / (oc + 0.5)
end

function score(m::CountModel, st::State, key::AgendaKey, c::UInt32)
  d = decisiondef(st, key)
  po = priorodds(m, priorkey(d, key[1], key[2], c))
  lpo = log(po)
  s = lpo
  for phi in d.features(st, key[3], c)
    s += log(smoothOdds(get(m.chosen, phi, 0.0), get(m.unchosen, phi, 0.0), po)) - lpo
  end
  s
end

waitscore(m::CountModel, st::State, key::AgendaKey) = log(priorodds(m, typekey(key[1], key[2])))

function learn!(m::CountModel, lname::Symbol, k::Int, labeled::Bool,
  candfeats::Vector{Tuple{UInt32,Vector{UInt64}}}, choice::UInt32)
  for (c, phis) in candfeats
    yes = c == choice
    d = yes ? m.chosen : m.unchosen
    for phi in phis
      d[phi] = get(d, phi, 0.0) + 1
    end
    for pk in unique((typekey(lname, k), labeled ? labelkey(lname, k, c) : typekey(lname, k)))
      (pc, poc) = get(m.prior, pk, (0.0, 0.0))
      m.prior[pk] = yes ? (pc + 1, poc) : (pc, poc + 1)
    end
  end
end

"""
Picks the gold choice, in left-to-right document order (round 1 of training).  A decision
whose gold answer doesn't exist yet (e.g. a head not yet tokenized) is blocked.
Gold lookups are specific to the default char/token layers; see oracle_choice in specs.jl.
"""
struct OracleModel <: AbstractModel end

function decide(::OracleModel, st::State, key::AgendaKey)
  d, i = decisiondef(st, key), key[3]
  d.ready(st, i) || return (NONE, -Inf)
  choice = oracle_choice(st, key)
  choice == PENDING && return (NONE, -Inf)
  pos = st.layers[key[1]].spec.anchor(st, i)
  (choice, -3.0 * pos - key[2] - (key[1] == first(st.order) ? 0 : 1))
end

"Training hook: records the features of every candidate as they looked when the decision was made."
mutable struct Recorder
  model::CountModel
  # (anchor, layer, decision index, labeled, candidate features, choice)
  buffer::Vector{Tuple{Int,Symbol,Int,Bool,Vector{Tuple{UInt32,Vector{UInt64}}},UInt32}}
end
Recorder(m::CountModel) = Recorder(m, [])

function observe!(r::Recorder, st::State, key::AgendaKey, choice::UInt32)
  d, i = decisiondef(st, key), key[3]
  cands = copy(d.candidates(st, i))
  choice in cands || push!(cands, choice)
  push!(r.buffer, (st.layers[key[1]].spec.anchor(st, i), key[1], key[2], d.labeled,
    [(c, d.features(st, i, c)) for c in cands], choice))
end

"Learn only from decisions inside closed sentences; the rest is re-processed with the next segment."
function segment_done!(r::Recorder, st::State)
  for (anchor, lname, k, labeled, cf, choice) in r.buffer
    anchor <= st.closed_through && learn!(r.model, lname, k, labeled, cf, choice)
  end
  empty!(r.buffer)
end
segment_discard!(r::Recorder) = empty!(r.buffer)

segment_done!(::Nothing, st) = nothing
segment_discard!(::Nothing) = nothing
