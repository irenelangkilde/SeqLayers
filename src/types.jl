# types.jl

# Reserved value codes, valid in every row of every grid
const NONE = UInt32(0)                 # absent: no such relation / no value
const PENDING = typemax(UInt32)        # not decided yet

abstract type Layer end
struct CharLayer <: Layer end
struct TokenLayer <: Layer end
struct SentenceLayer <: Layer end

abstract type AbstractState end

# (state, column) -> T
const PosFn{T} = FunctionWrapper{T,Tuple{AbstractState,Int}}
# (state, column, candidate) -> feature keys for choosing that candidate
const PhiFn = FunctionWrapper{Vector{UInt64},Tuple{AbstractState,Int,UInt32}}

"A feature row: its value for a column is computed once, when the column is created."
struct FeatDef
  name::Symbol
  value::PosFn{UInt32}
end
FeatDef(name::Symbol, f::Function) = FeatDef(name, PosFn{UInt32}(f))

"""
A relation row: pointers from columns of this layer to columns of layer `to`
(NONE = no such relation, PENDING = not decided yet).  Static relations are set once
at column creation; dynamic ones are recomputed after every commit.
"""
struct RelDef
  name::Symbol
  to::Symbol
  dynamic::Bool
  target::PosFn{UInt32}
end
RelDef(name::Symbol, to::Symbol, dynamic::Bool, f::Function) = RelDef(name, to, dynamic, PosFn{UInt32}(f))

"A path through relations ending at a feature, e.g. FeatPath([:left_neighbor, :token_of], :form)."
struct FeatPath
  rels::Vector{Symbol}
  feat::Symbol
end
FeatPath(feat::Symbol) = FeatPath(Symbol[], feat)

const ReadCell = Tuple{Symbol,Int,Int,UInt32}

"""
Something decided for each column of a layer.  The chosen candidate is written to the
decision's own row.  `ready` is false while prerequisites are missing (the decision then
waits on the agenda).  `open` means more candidates may appear later (e.g. heads not yet
tokenized), so a lone candidate is not committed automatically.  `labeled` means candidates
come from a fixed label set (gap labels, relations), so each label gets its own prior;
otherwise (e.g. head positions) all candidates share one prior.

`features` gives ready-made feature keys (conjunctions included) for CountModel; `items`
gives the single, unconjoined features that LatticeModel and ChainModel combine themselves,
one per slot, with ABSENT for a feature whose value is still pending.  For labeled decisions,
items describe the context only (the label is handled by the model).

For training from the gold state (omnidirectional): `allcandidates` is what the decision
chooses among before any constraint narrows it, and `mask(st, key)` hides the decision and
whatever is derived from it, returning a function that restores the state.
"""
struct DecisionDef
  name::Symbol
  candidates::PosFn{Vector{UInt32}}
  features::PhiFn
  ready::PosFn{Bool}
  open::Bool
  labeled::Bool
  items::PhiFn
  allcandidates::PosFn{Vector{UInt32}}
  mask::Function
end
function DecisionDef(name::Symbol; candidates::Function, features::Function,
  ready::Function = (st, i) -> true, open::Bool = false, labeled::Bool = true,
  items::Function = (st, i, c) -> UInt64[], allcandidates::Function = candidates,
  mask::Function = default_mask)
  DecisionDef(name, PosFn{Vector{UInt32}}(candidates), PhiFn(features), PosFn{Bool}(ready), open, labeled,
    PhiFn(items), PosFn{Vector{UInt32}}(allcandidates), mask)
end

"Item for a feature whose value isn't known yet: models treat it as absent, not as a value."
const ABSENT = UInt64(0)

"""
Runs after each commit.  For a commit in its own layer it gets that column; for a
commit in another layer it gets 0.  When `condition` holds, `generate!` adds structure,
usually columns in the layer above.
"""
struct Constraint
  name::Symbol
  condition::PosFn{Bool}
  generate!::PosFn{Nothing}
end
Constraint(name::Symbol, condition::Function, generate!::Function) =
  Constraint(name, PosFn{Bool}(condition), PosFn{Nothing}((st, i) -> (generate!(st, i); nothing)))

struct LayerSpec{L<:Layer}
  name::Symbol
  feats::Vector{FeatDef}
  rels::Vector{RelDef}
  decisions::Vector{DecisionDef}
  constraints::Vector{Constraint}
  context::Vector{FeatPath}          # what the model may look at from a column of this layer
  anchor::PosFn{Int}                 # column -> position in the base (char) layer
  rows::Dict{Symbol,Int}             # row name -> grid row
  relto::Dict{Symbol,Symbol}         # relation name -> target layer
end

function LayerSpec{L}(name, feats, rels, decisions, constraints, context, anchor::Function) where {L}
  names = [map(f -> f.name, feats); map(r -> r.name, rels); map(d -> d.name, decisions)]
  allunique(names) || error("duplicate row names in layer $name: $names")
  LayerSpec{L}(name, feats, rels, decisions, constraints, context, PosFn{Int}(anchor),
    Dict(n => k for (k, n) in enumerate(names)), Dict(r.name => r.to for r in rels))
end

"One layer's grid: rows = features, relations and decisions; columns = positions."
mutable struct Sequence{L<:Layer}
  spec::LayerSpec{L}
  grid::Matrix{UInt32}
  ncols::Int
end
Sequence(spec::LayerSpec{L}, capacity::Int) where {L} =
  Sequence{L}(spec, fill(PENDING, length(spec.rows), capacity), 0)

const CharSeq = Sequence{CharLayer}
const TokenSeq = Sequence{TokenLayer}
const SentenceSeq = Sequence{SentenceLayer}

const READ_NCOLS_ROW = 0
const READ_NCOLS_COL = 0
const READ_TRACE = Ref{Union{Nothing,Vector{ReadCell}}}(nothing)

@inline function track_read(s::Sequence, row::Int, i::Int, v::UInt32)
  trace = READ_TRACE[]
  trace === nothing || push!(trace, (s.spec.name, row, i, v))
  v
end

function Base.getindex(s::Sequence, row::Symbol, i::Integer)
  rowi = s.spec.rows[row]
  track_read(s, rowi, Int(i), s.grid[rowi, i])
end
Base.setindex!(s::Sequence, v, row::Symbol, i::Integer) = (s.grid[s.spec.rows[row], i] = UInt32(v))
function Base.length(s::Sequence)
  n = s.ncols
  trace = READ_TRACE[]
  trace === nothing || push!(trace, (s.spec.name, READ_NCOLS_ROW, READ_NCOLS_COL, UInt32(n)))
  n
end

"Open-vocabulary value codes (e.g. word forms), shared across segments and documents."
struct Lexicon
  vals::Dict{Symbol,Dict{String,UInt32}}
  names::Dict{Symbol,Vector{String}}
end
Lexicon() = Lexicon(Dict{Symbol,Dict{String,UInt32}}(), Dict{Symbol,Vector{String}}())

function intern!(lex::Lexicon, feat::Symbol, s::AbstractString)
  d = get!(() -> Dict{String,UInt32}(), lex.vals, feat)
  names = get!(() -> String[], lex.names, feat)
  get!(d, s) do
    push!(names, s)
    UInt32(length(names))
  end
end

"Gold annotation in document character positions."
struct Gold
  gap::Vector{UInt32}                            # per char: gold gap label (NONE for whitespace)
  head::Dict{UnitRange{Int},UnitRange{Int}}      # token span -> head span (0:-1 for root)
  deprel::Dict{UnitRange{Int},UInt32}            # token span -> relation code
  sentends::Set{Int}                             # last char of each sentence
  tags::Dict{Symbol,Dict{UnitRange{Int},UInt32}} # tag decision -> token span -> label code
end
Gold(gap, head, deprel, sentends) = Gold(gap, head, deprel, sentends, Dict{Symbol,Dict{UnitRange{Int},UInt32}}())

struct Doc
  text::Vector{Char}
  gold::Union{Gold,Nothing}
end

const AgendaKey = Tuple{Symbol,Int,Int}          # (layer, decision index, column)

"Solver state for one segment of a document."
mutable struct State <: AbstractState
  doc::Doc
  span::UnitRange{Int}                           # doc char range of this segment
  final::Bool                                    # segment ends the document
  edge::Int                                      # chars at the end kept frozen unless final
  layers::Dict{Symbol,Sequence}
  order::Vector{Symbol}                          # layers bottom-up; the first is the base layer
  lex::Lexicon
  agenda::PriorityQueue{AgendaKey,Float64}       # pending decisions, highest margin first
  best::Dict{AgendaKey,UInt32}                   # current best candidate per pending decision
  fresh::Vector{AgendaKey}                       # decisions created since the last agenda update
  reads::Dict{AgendaKey,Vector{ReadCell}}         # grid cells read the last time a decision was scored
  closed_through::Int                            # last base-layer column inside a closed sentence
  hidden::Set{Tuple{Symbol,Int}}                 # columns masked while recording training events
  infer::Bool                                    # constraints may infer decisions (not for the oracle)
  memo::Union{Nothing,Dict{Symbol,Any}}          # shared by dynamic relations during one refresh! (else nothing)
end
