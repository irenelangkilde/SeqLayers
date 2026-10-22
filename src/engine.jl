# engine.jl  --  task-independent: columns, relations, context, agenda loop

function State(doc::Doc, span::UnitRange{Int}, final::Bool, edge::Int, specs::Vector{<:LayerSpec}, lex::Lexicon)
  n = length(span)
  layers = Dict{Symbol,Sequence}(s.name => Sequence(s, max(n, 1)) for s in specs)
  st = State(doc, span, final, edge, layers, [s.name for s in specs], lex,
    PriorityQueue{AgendaKey,Float64}(Base.Order.Reverse), Dict{AgendaKey,UInt32}(), AgendaKey[],
    Dict{AgendaKey,Vector{ReadCell}}(), 0, Set{Tuple{Symbol,Int}}(), true, nothing)
  base = first(st.order)
  for _ in 1:n
    addcol!(st, base)
  end
  st
end

decisiondef(st::State, key::AgendaKey) = st.layers[key[1]].spec.decisions[key[2]]

"Append a column to layer `lname`.  `init` sets rows directly; other rows come from their defs."
function addcol!(st::State, lname::Symbol, init::Pair{Symbol,<:Integer}...)
  seq = st.layers[lname]
  i = seq.ncols += 1
  if i > size(seq.grid, 2)
    seq.grid = hcat(seq.grid, fill(PENDING, size(seq.grid)...))
  end
  given = Set{Symbol}()
  for (r, v) in init
    seq[r, i] = v
    push!(given, r)
  end
  for r in seq.spec.rels   # relations before features: features may read them
    r.name in given || (seq[r.name, i] = r.target(st, i))
  end
  for f in seq.spec.feats
    f.name in given || (seq[f.name, i] = f.value(st, i))
  end
  for (k, d) in enumerate(seq.spec.decisions)
    seq[d.name, i] = PENDING
    push!(st.fresh, (lname, k, i))
  end
  i
end

"""
Follow `path` from column i of layer `lname`; NONE or PENDING if the path breaks.  Landing on
a hidden (masked) column reads as PENDING.
"""
function follow(st::State, lname::Symbol, i::Int, path::FeatPath)
  for r in path.rels
    seq = st.layers[lname]
    v = seq[r, i]
    (v == NONE || v == PENDING) && return v
    lname = seq.spec.relto[r]
    i = Int(v)
    isempty(st.hidden) || (lname, i) in st.hidden && return PENDING
  end
  st.layers[lname][path.feat, i]
end

"""
Default mask for recording a decision from the gold state: hide its own value (dynamic
relations that read it are refreshed) and return a function that restores it.
"""
function default_mask(st::State, key::AgendaKey)
  lname, k, i = key
  seq = st.layers[lname]
  name = seq.spec.decisions[k].name
  old = seq[name, i]
  seq[name, i] = PENDING
  refresh!(st)
  () -> (seq[name, i] = old; refresh!(st))
end

context(st::State, lname::Symbol, i::Int) = [follow(st, lname, i, p) for p in st.layers[lname].spec.context]

"""
Features for choosing label c at column i: every context value, and every pair of
context values, conjoined with c.  `tag` names the decision.
"""
function label_features(st::State, lname::Symbol, tag::UInt64, i::Int, c::UInt32)
  ctx = context(st, lname, i)
  phi = UInt64[]
  for a in eachindex(ctx)
    push!(phi, hash((tag, a, ctx[a], c)))
    for b in a+1:lastindex(ctx)
      push!(phi, hash((tag, a, ctx[a], b, ctx[b], c)))
    end
  end
  phi
end

"""
Features for linking column i to column c of the same layer (c == NONE: no link):
pairs of context values (dependent x candidate) over `paths`, plus a distance bucket.
"""
function link_features(st::State, lname::Symbol, tag::UInt64, i::Int, c::UInt32, dist::Int, paths)
  ps = st.layers[lname].spec.context
  dv = [follow(st, lname, i, ps[k]) for k in paths]
  hv = c == NONE ? fill(NONE, length(paths)) : [follow(st, lname, Int(c), ps[k]) for k in paths]
  phi = UInt64[hash((tag, 0, dist))]
  for a in eachindex(dv), b in eachindex(hv)
    push!(phi, hash((tag, a, dv[a], b, hv[b], sign(dist))))
  end
  phi
end

"Item for slot value v, or ABSENT if v isn't known yet."
item(v::UInt32, key) = v == PENDING ? ABSENT : hash(key)

"Single items for a labeled decision: each context value, tagged with its path (ABSENT if pending)."
label_items(st::State, lname::Symbol, tag::UInt64, i::Int) =
  [item(v, (tag, a, v)) for (a, v) in enumerate(context(st, lname, i))]

"""
Single items for linking column i to column c (c == NONE: no link): the dependent's and the
candidate's values over `paths`, the distance bucket and the direction.  With `rootabsent`, the
no-link candidate's values are absent (it has no token), so only the distance and direction mark it.
"""
function link_items(st::State, lname::Symbol, tag::UInt64, i::Int, c::UInt32, dist::Int, paths; rootabsent = false)
  ps = st.layers[lname].spec.context
  items = UInt64[hash((tag, 3, dist)), hash((tag, 4, sign(dist)))]
  for (a, k) in enumerate(paths)
    dv = follow(st, lname, i, ps[k])
    hv = c == NONE ? NONE : follow(st, lname, Int(c), ps[k])
    push!(items, item(dv, (tag, 1, a, dv)))
    push!(items, rootabsent && c == NONE ? ABSENT : item(hv, (tag, 2, a, hv)))
  end
  items
end

"Recompute every dynamic relation row."
function refresh!(st::State)
  st.memo = Dict{Symbol,Any}()          # dynamic relations may share work (e.g. children lists) during this refresh
  try
  for lname in st.order
    seq = st.layers[lname]
    for r in seq.spec.rels
      r.dynamic || continue
      row = seq.spec.rows[r.name]
      for i in 1:seq.ncols
        seq.grid[row, i] = r.target(st, i)
      end
    end
  end
  finally
    st.memo = nothing
  end
end

"Put newly created decisions on the agenda; ones with no candidates are settled as NONE."
function activate!(st::State)
  for key in st.fresh
    d = decisiondef(st, key)
    if isempty(d.candidates(st, key[3]))
      st.layers[key[1]][d.name, key[3]] = NONE
    else
      st.agenda[key] = 0.0
    end
  end
  empty!(st.fresh)
end

@inline function read_changed(st::State, cell::ReadCell)
  lname, row, i, old = cell
  seq = st.layers[lname]
  v = row == READ_NCOLS_ROW ? UInt32(seq.ncols) : seq.grid[row, i]
  v != old
end

function compact_reads!(reads::Vector{ReadCell})
  length(reads) < 2 && return reads
  seen = Set{Tuple{Symbol,Int,Int}}()
  compact = ReadCell[]
  sizehint!(compact, length(reads))
  for (lname, row, i, v) in reads
    id = (lname, row, i)
    id in seen && continue
    push!(seen, id)
    push!(compact, (lname, row, i, v))
  end
  compact
end

function tracked_decide(model, st::State, key::AgendaKey)
  reads = ReadCell[]
  old = READ_TRACE[]
  READ_TRACE[] = reads
  try
    choice, margin = decide(model, st, key)
    st.reads[key] = compact_reads!(reads)
    return choice, margin
  finally
    READ_TRACE[] = old
  end
end

function rescore!(st::State, model)
  for key in collect(keys(st.agenda))
    reads = get(st.reads, key, nothing)
    if reads !== nothing && haskey(st.best, key) && !any(cell -> read_changed(st, cell), reads)
      continue
    end
    choice, margin = tracked_decide(model, st, key)
    st.agenda[key] = margin
    st.best[key] = choice
  end
end

"""
Easy-first greedy search: repeatedly commit the most confident ready decision in any
layer, then refresh relations and run constraints (which may add columns and decisions).
Stops when the agenda is empty or everything left is blocked/frozen.
"""
function solve!(st::State, model; observer = nothing)
  while true
    activate!(st)
    isempty(st.agenda) && break
    rescore!(st, model)
    key, margin = first(st.agenda)
    margin == -Inf && break
    delete!(st.agenda, key)
    choice = st.best[key]
    delete!(st.best, key)
    delete!(st.reads, key)
    observer === nothing || observe!(observer, st, key, choice)
    commit!(st, key, choice)
  end
  st
end

function commit!(st::State, key::AgendaKey, choice::UInt32)
  lname, k, i = key
  seq = st.layers[lname]
  seq[seq.spec.decisions[k].name, i] = choice
  refresh!(st)
  for l in st.order
    for c in st.layers[l].spec.constraints
      j = l == lname ? i : 0
      c.condition(st, j) && c.generate!(st, j)
    end
  end
  refresh!(st)
end
