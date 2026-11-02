def edit(p, pairs):
    t = open(p).read()
    for old, new in pairs:
        assert t.count(old) == 1, (p, old[:80], t.count(old)); t = t.replace(old, new)
    open(p, "w").write(t)

edit("src/specs.jl", [
 # label set and gold
 ('''tagcode(name::Symbol, s::AbstractString) =''', '''# oclass (default_specs(oclass = true); user's design 2026-11-02): open-class words (NOUN, PROPN, VERB, ADJ, ADV) vs
# closed-class (everything else), a tag decision of its own, from gold UPOS
const OCLASS_LABELS = ["open", "closed"]
const OCLASS_CODES = UInt32[1, 2]
const OCLASS_TAG = hash("token.oclass")
const OPEN_UPOS = ("NOUN", "PROPN", "VERB", "ADJ", "ADV")
oclass_code(upos::AbstractString) = upos in OPEN_UPOS ? UInt32(1) : UInt32(2)
tagcode(name::Symbol, s::AbstractString) ='''),
 # oracle
 ('''  if haskey(TAGSETS, name) || haskey(ALLHEADSETS, name)
    fallback =''', '''  if name == :oclass
    return haskey(gold.tags, :oclass) ? get(gold.tags[:oclass], span, UInt32(2)) : UInt32(2)
  end
  if haskey(TAGSETS, name) || haskey(ALLHEADSETS, name)
    fallback ='''),
 # BIOES items, after the kinship items
 ('''"Word classes for the head_wc properties''', '''const BIOES_TAG = hash("token.attach.bioes")

"""
BIOES label of token u by its own subtree (user's design 2026-11-02), with t's attachment taken to be c: 1 S (no
dependents), 2 B (dependents only to its right: it starts its subtree), 3 E (only to its left: it ends it), 4 I (both
sides).  PENDING while a side has no dependent yet and some other token on it (in u's sentence) is unattached.
"""
function bioes_label(st, u::Int, t::Int, c)
  toks = st.layers[:token]
  ord, rank, cp, sfirst, slast = relranks(st)
  ru, rt = rank[u], rank[t]
  ch = children_index(st)[u]
  hasL = any(v -> v != t && rank[v] < ru, ch) || (c == u && rt < ru)
  hasR = any(v -> v != t && rank[v] > ru, ch) || (c == u && rt > ru)
  function settled(a, b)                             # every token in ranks a..b attached (t: hypothetically)
    a > b && return true
    n = cp[b+1] - cp[a]
    a <= rt <= b && toks[:attach, t] == PENDING && (n -= 1)
    n == 0
  end
  open_ = toks[:sentence, u] == PENDING
  okL = hasL || (!open_ && settled(sfirst[ru], ru - 1))
  okR = hasR || (!open_ && settled(ru + 1, slast[ru]))
  (okL && okR) || return PENDING
  UInt32(hasL ? (hasR ? 4 : 3) : (hasR ? 2 : 1))
end

"""
BIOES items for t under candidate head c (default_specs(bioes = true)): the gap to t's left as the pair (label of t's
left neighbour, t's label), the gap to its right (t's, the right neighbour's), and c's own label -- all with t attached
to c.  Labels 1-4 as bioes_label, 5 no neighbour in the sentence; c the root: 6.
"""
function bioes_values(st, t, c)
  toks = st.layers[:token]
  c == PENDING && return (PENDING, PENDING, PENDING)
  lab(u) = u == PENDING ? PENDING : u == NONE ? UInt32(5) :
    (toks[:sentence, t] == PENDING || toks[:sentence, Int(u)] == PENDING) ? PENDING :
    toks[:sentence, Int(u)] != toks[:sentence, t] ? UInt32(5) : bioes_label(st, Int(u), t, c)
  lt = bioes_label(st, t, t, c)
  ll, lr = lab(adjacent_token(st, t, -1)), lab(adjacent_token(st, t, 1))
  pair(a, b) = (a == PENDING || b == PENDING) ? PENDING : UInt32(5 * (a - 1) + b)
  lc = c == NONE ? UInt32(6) : bioes_label(st, Int(c), t, c)
  (pair(ll, lt), pair(lt, lr), lc)
end
bioes_items(st, t, c) = UInt64[item(v, (BIOES_TAG, j, v)) for (j, v) in enumerate(bioes_values(st, t, c))]

"Word classes for the head_wc properties'''),
 # default_specs
 ('''  kinitems::Bool = false, headkin::Vector{Symbol} = Symbol[], neighbor2::Bool = false)''',
  '''  kinitems::Bool = false, headkin::Vector{Symbol} = Symbol[], neighbor2::Bool = false, oclass::Bool = false, bioes::Bool = false)'''),
 ('''  append!(linkpaths, nn2at .+ eachindex(nn2paths))
''', '''  append!(linkpaths, nn2at .+ eachindex(nn2paths))
  # oclass: the token's, its neighbours' and its head's open / closed class, last in the context; the first three in the
  # attachment features (both sides), as upos
  oclpaths = oclass ? [FeatPath(:oclass), FeatPath([:left_neighbor], :oclass), FeatPath([:right_neighbor], :oclass),
    FeatPath([:head], :oclass)] : FeatPath[]
  oclat = nn2at + length(nn2paths)
  oclass && append!(linkpaths, oclat .+ (1:3))
'''),
 ('''          kinitems ? kin_items(st, t, c) : UInt64[],
''', '''          kinitems ? kin_items(st, t, c) : UInt64[],
          bioes ? bioes_items(st, t, c) : UInt64[],
'''),
 ('''      [FeatPath(n) for n in headkin],
      nn2paths),''', '''      [FeatPath(n) for n in headkin],
      nn2paths,
      oclpaths),'''),
 ('''+ length(headkin) + length(nn2paths) || error''', '''+ length(headkin) + length(nn2paths) + length(oclpaths) || error'''),
 ('''      [DecisionDef(wnames[l];''', '''      [DecisionDef(:oclass;
          candidates = (st, t) -> OCLASS_CODES,
          features = (st, t, c) -> label_features(st, :token, OCLASS_TAG, t, c),
          items = (st, t, c) -> label_items(st, :token, OCLASS_TAG, t)) for _ in 1:oclass],
      [DecisionDef(wnames[l];'''),
])
edit("src/wordclass.jl", [
 ('''  isgen(f) = f in (:upos, :xpos, :basepos) || iswclass(f)''', '''  isgen(f) = f in (:upos, :xpos, :basepos, :oclass) || iswclass(f)'''),
])
edit("src/conllu.jl", [
 ('''  tags = Dict(name => Dict{UnitRange{Int},UInt32}() for name in [collect(keys(TAGSETS)); collect(keys(ALLHEADSETS))])''',
  '''  tags = Dict(name => Dict{UnitRange{Int},UInt32}() for name in [collect(keys(TAGSETS)); collect(keys(ALLHEADSETS)); :oclass])'''),
 ('''      for (name, code) in pairs(gold_tags(w.form, w.upos, w.xpos))
        tags[name][s] = code
      end''', '''      for (name, code) in pairs(gold_tags(w.form, w.upos, w.xpos))
        tags[name][s] = code
      end
      tags[:oclass][s] = oclass_code(w.upos)'''),
])
edit("experiments/dev_eval.jl", [
 ('#   neighbor2    true:', '''#   oclass       true: an open / closed class tag decision (open: NOUN, PROPN, VERB, ADJ, ADV), in the context (own,
#                neighbours', head's), the attachment features and, as the coarsest rung, the backoff ladders
#   bioes        true: three attachment items per candidate c -- the token's subtree BIOES label (S none, B right
#                dependents only, E left only, I both) paired with each neighbour's across the two gaps, and c's,
#                all with the token attached to c
#   neighbor2    true:'''),
 ('"neighbor2" => "false",', '"neighbor2" => "false", "oclass" => "false", "bioes" => "false",'),
 ('''    kinitems = cfg["kinitems"] == "true", headkin = headkin_names(cfg), neighbor2 = cfg["neighbor2"] == "true")''',
  '''    kinitems = cfg["kinitems"] == "true", headkin = headkin_names(cfg), neighbor2 = cfg["neighbor2"] == "true",
    oclass = cfg["oclass"] == "true", bioes = cfg["bioes"] == "true")'''),
 ('''* (cfg["neighbor2"] == "true" ? "-nn2" : ""))''', '''* (cfg["neighbor2"] == "true" ? "-nn2" : "") *
    (cfg["oclass"] == "true" ? "-ocl" : "") * (cfg["bioes"] == "true" ? "-bioes" : ""))'''),
 ('''(cfg["neighbor2"] == "true" ? ";neighbor2" : "") * (cfg["goldtok"] == "true" ? ";goldtok" : "") * "]"''',
  '''(cfg["neighbor2"] == "true" ? ";neighbor2" : "") * (cfg["oclass"] == "true" ? ";oclass" : "") *
      (cfg["bioes"] == "true" ? ";bioes" : "") * (cfg["goldtok"] == "true" ? ";goldtok" : "") * "]"'''),
 ('''      kinitems = $(cfg["kinitems"] == "true"), headkin = $(headkin_names(cfg)), neighbor2 = $(cfg["neighbor2"] == "true"))''',
  '''      kinitems = $(cfg["kinitems"] == "true"), headkin = $(headkin_names(cfg)), neighbor2 = $(cfg["neighbor2"] == "true"),
      oclass = $(cfg["oclass"] == "true"), bioes = $(cfg["bioes"] == "true"))'''),
])
print("ok")
