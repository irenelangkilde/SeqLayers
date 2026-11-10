# wordclass.jl  --  nested word classes as token decisions
#
# Classes come from juliaNLP/clusters (classes.jl: nested k-means of the words' linear context
# vectors, levels of 30000, 3000, 300 and 30 classes), exported as wordclasses.tsv.  Each level is
# a token-layer decision (:wclass30000, ...) with labels 1..k for the classes and k+1 for
# "unknown".  Candidate sets are small: a word in the table has only its own class; any other
# word gets "unknown" (its gold value) plus the most common classes among table words sharing its
# longest suffix, so a model could guess one.  The decisions' values reach the context like tags.

"""
Word classes at several levels.  `class[w, l]` is word w's class at level l; `suffix[l]` maps a
suffix to the classes (most common first, by word frequency) of the table words ending with it.
"""
struct WordClasses
  sizes::Vector{Int}
  index::Dict{String,Int}
  class::Matrix{UInt32}
  suffix::Vector{Dict{String,Vector{UInt32}}}
  suffixlens::Vector{Int}                     # longest first
end

const WCLASS_SUFFIXES = (5, 4, 3)

"""
Read wordclasses.tsv (a `#sizes` header, then form, frequency, class per level).  Words shorter
than a suffix length plus 2 don't vote for that suffix; each suffix keeps its `nsuffix` most
common classes.
"""
function load_wordclasses(file::AbstractString; nsuffix::Int = 3, suffixlens = WCLASS_SUFFIXES)
  isdir(file) && (file = joinpath(file, "wordclasses.tsv"))
  lines = eachline(file)
  head, _ = iterate(lines)
  startswith(head, "#sizes") || error("$file: no #sizes header")
  sizes = parse.(Int, split(head, '\t')[2:end])
  forms, freqs, rows = String[], Int[], Vector{UInt32}[]
  for l in lines
    f = split(l, '\t')
    push!(forms, f[1]); push!(freqs, parse(Int, f[2])); push!(rows, parse.(UInt32, f[3:end]))
  end
  class = permutedims(reduce(hcat, rows))
  votes = [Dict{String,Dict{UInt32,Int}}() for _ in sizes]
  for (w, form) in enumerate(forms), n in suffixlens
    length(form) >= n + 2 || continue
    s = last(form, n)
    for l in eachindex(sizes)
      v = get!(() -> Dict{UInt32,Int}(), votes[l], s)
      v[class[w, l]] = get(v, class[w, l], 0) + freqs[w]
    end
  end
  suffix = [Dict(s => first.(sort!(collect(v); by = p -> (-p.second, p.first)))[1:min(end, nsuffix)] for (s, v) in vl)
    for vl in votes]
  WordClasses(sizes, Dict(f => w for (w, f) in enumerate(forms)), class, suffix, sort(collect(suffixlens); rev = true))
end

wclass_name(k::Int) = Symbol(:wclass, k)
wclass_unknown(wc::WordClasses, l::Int) = UInt32(wc.sizes[l] + 1)

"Candidates for (lowercased) `form` at level l: its class, or unknown plus classes of its longest known suffix."
function wclass_candidates(wc::WordClasses, l::Int, form::AbstractString)
  w = get(wc.index, form, 0)
  w > 0 && return UInt32[wc.class[w, l]]
  cands = UInt32[wclass_unknown(wc, l)]
  for n in wc.suffixlens
    length(form) >= n + 1 || continue
    cs = get(wc.suffix[l], last(form, n), nothing)
    cs === nothing && continue
    append!(cands, cs)
    break
  end
  cands
end

"""
Fill in the gold head_wc labels (the gold head word's class at each level in `levels`, or root) of documents read
by read_conllu, once the classes are loaded.
"""
function add_head_wclass_gold!(docs, wc::WordClasses, levels)
  for d in docs, n in levels
    l = findfirst(==(n), wc.sizes)
    name = Symbol(:head_wc, n)
    tags = get!(() -> Dict{UnitRange{Int},UInt32}(), d.gold.tags, name)
    root = UInt32(length(ALLHEADSETS[name]))
    for (s, hs) in d.gold.head
      tags[s] = isempty(hs) ? root : wclass_gold(wc, l, lowercase(String(d.text[hs])))
    end
  end
  docs
end

"Gold class of `form` at level l: its class, or unknown."
wclass_gold(wc::WordClasses, l::Int, form::AbstractString) =
  (w = get(wc.index, form, 0); w > 0 ? wc.class[w, l] : wclass_unknown(wc, l))

"""
Slots of the word-class values in each token decision's records (attachment: the dependent's and
the candidate's value of each class path; label decisions: the class context paths), for
ChainModel's `noleft` -- classes in chains only, not as leftovers.
"""
function wclass_slots(specs; levels = nothing)
  toks = specs[2]
  nlev = count(d -> iswclass(d.name), toks.decisions)
  nlev == 0 && return Dict{UInt64,Set{Int}}()
  ls = levels === nothing ? (1:nlev) : [l for l in levels if 1 <= l <= nlev]   # (levels: indices into the class sizes)
  paths = 22 .+ ls                                       # the class context paths (see default_specs)
  a0 = length(LINK_PATHS)                                # their positions among the link paths follow LINK_PATHS
  out = Dict{UInt64,Set{Int}}()
  for (k, d) in enumerate(toks.decisions)
    out[typekey(:token, k)] = k == 1 ? Set([s for l in ls for s in (2(a0 + l) + 1, 2(a0 + l) + 2)]) : Set(paths)
  end
  out
end

"oclass in the generalization ladders (oclassladder; false keeps it a feature and attachment item only -- 2026-11-10)."
const OCLASS_LADDER = Ref(true)

"""
For each token decision type, each form slot's generalization slots on the same token (same relation
path: the token itself, a neighbour, ...): its upos, xpos, basepos, suffixes and word classes, when
present.  Attachment: per side (dependent, candidate), over the link paths; label decisions: over the
context paths.  For the soft redundant pool (ChainOpts.poolsigma).
"""
function generalization_slots(specs)
  toks = specs[2]
  ctx = toks.context
  isgen(f) = f in (:upos, :xpos, :basepos) || (f == :oclass && OCLASS_LADDER[]) || iswclass(f) || startswith(String(f), "suffix")
  function group(paths, slotof)
    forms, gens = Dict{Vector{Symbol},Int}(), Dict{Vector{Symbol},Vector{Int}}()
    for (j, k) in enumerate(paths)
      p = ctx[k]
      p.feat == :form && (forms[p.rels] = slotof(j))
      isgen(p.feat) && push!(get!(() -> Int[], gens, p.rels), slotof(j))
    end
    Dict(s => gens[r] for (r, s) in forms if haskey(gens, r))
  end
  out = Dict{UInt64,Dict{Int,Vector{Int}}}()
  lp = get(LINKPATHS, toks, LINK_PATHS)
  for (k, d) in enumerate(toks.decisions)
    out[typekey(:token, k)] = k == 1 ? merge(group(lp, j -> 2j + 1), group(lp, j -> 2j + 2)) : group(eachindex(ctx), j -> j)
  end
  out
end

"Is this a word-class decision?"
iswclass(name::Symbol) = startswith(String(name), "wclass")
