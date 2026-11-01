p = "src/specs.jl"; s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)
# label sets
rep('''  head_sibin = ["none", "1", "2", "3-5", "6-10", "11+", "root"], head_sibout = ["none", "1", "2", "3-5", "6-10", "11+", "root"], head_sibleft = ["none", "1", "2", "3-5", "6-10", "11+", "root"], head_sibright = ["none", "1", "2", "3-5", "6-10", "11+", "root"]))''',
'''  head_sibin = ["none", "1", "2", "3-5", "6-10", "11+", "root"], head_sibout = ["none", "1", "2", "3-5", "6-10", "11+", "root"], head_sibleft = ["none", "1", "2", "3-5", "6-10", "11+", "root"], head_sibright = ["none", "1", "2", "3-5", "6-10", "11+", "root"],
  # head_kinleft / head_kinright (default_specs(headkin = [...])): the token's kinship to its left / right neighbour
  # token in the tree its head gives (see KIN_LABELS), or root
  head_kinleft = ["parent", "dep", "sibling", "aunt/uncle", "niece/nephew", "1stcousin", "2ndcousin", "grandparent",
    "other_ancestor", "other_descendant", "other_relative", "none", "root"],
  head_kinright = ["parent", "dep", "sibling", "aunt/uncle", "niece/nephew", "1stcousin", "2ndcousin", "grandparent",
    "other_ancestor", "other_descendant", "other_relative", "none", "root"]))''')
rep('''"Word classes for the head_wc properties''', '''"Kinship head properties -> the neighbour's direction."
const HEADKIN_DIR = (head_kinleft = -1, head_kinright = 1)

"""
Kinship of a token t to a neighbour n (user's design 2026-11-01), from t's side: n is t's parent (head), dep
(dependent), sibling, aunt/uncle (a sibling of t's parent), niece/nephew (a child of t's sibling), 1st / 2nd cousin,
grandparent, other ancestor, other descendant (a grandchild or below), other relative (any other link through a
common ancestor, or none in the tree); none: no neighbour in the sentence.  Codes are positions in this list.
"""
const KIN_LABELS = ["parent", "dep", "sibling", "aunt/uncle", "niece/nephew", "1stcousin", "2ndcousin", "grandparent",
  "other_ancestor", "other_descendant", "other_relative", "none"]
const KIN_NONE, KIN_OTHER = UInt32(12), UInt32(11)

"The kinship code for i steps up from t and j steps up from n to their lowest common ancestor."
function kincode(i::Int, j::Int)
  lab = (i, j) == (1, 0) ? "parent" : (i, j) == (0, 1) ? "dep" : i == 0 ? "other_descendant" :
    (i, j) == (2, 0) ? "grandparent" : j == 0 ? "other_ancestor" : (i, j) == (1, 1) ? "sibling" :
    (i, j) == (2, 1) ? "aunt/uncle" : (i, j) == (1, 2) ? "niece/nephew" : (i, j) == (2, 2) ? "1stcousin" :
    (i, j) == (3, 3) ? "2ndcousin" : "other_relative"
  UInt32(findfirst(==(lab), KIN_LABELS))
end

"""
Kinship code of t to n with t's head taken to be c (NONE: root) and every other token's head from `headof` (a token
index, NONE or PENDING): PENDING while an attachment the answer needs is unknown; other_relative when the two chains
end at different roots; PENDING too on a cycle (c under t).
"""
function kin_between(headof, t::Int, n::Int, c, maxn::Int)
  up = Dict{Int,Int}(t => 0)                         # t's ancestors (c first) and their steps from t
  h, d, complete = c, 0, true
  while true
    h == PENDING && (complete = false; break)
    h == NONE && break
    u = Int(h)
    haskey(up, u) && return PENDING
    d += 1; up[u] = d
    d > maxn && return PENDING
    h = headof(u)
  end
  v, j = n, 0
  while true
    haskey(up, v) && return kincode(up[v], j)
    hv = headof(v)
    hv == PENDING && return PENDING
    hv == NONE && return complete ? KIN_OTHER : PENDING
    v = Int(hv); j += 1
    j > maxn && return PENDING
  end
end

"t's kinship code to its neighbour on side `dir` (-1 left, +1 right) if attached to c: PENDING while unknown."
function kin_value(st, t, c, dir)
  c == PENDING && return PENDING
  toks = st.layers[:token]
  n = adjacent_token(st, t, dir)
  n == PENDING && return PENDING
  n == NONE && return KIN_NONE
  st_, sn = toks[:sentence, t], toks[:sentence, Int(n)]
  (st_ == PENDING || sn == PENDING) && return PENDING
  st_ == sn || return KIN_NONE
  kin_between(u -> toks[:attach, u], t, Int(n), c, length(toks))
end
const KIN_TAG = hash("token.attach.kin")
"Kinship items for t under candidate head c (default_specs(kinitems = true)): to its left neighbour, then its right."
kin_items(st, t, c) = UInt64[item(v, (KIN_TAG, j, v)) for (j, v) in enumerate((kin_value(st, t, c, -1), kin_value(st, t, c, 1)))]

"Word classes for the head_wc properties''')
# gold values
rep('''    head_sibleft = headcode(:head_sibleft, "root"), head_sibright = headcode(:head_sibright, "root"))
  lo, hi = minmax(i, h)''', '''    head_sibleft = headcode(:head_sibleft, "root"), head_sibright = headcode(:head_sibright, "root"),
    head_kinleft = headcode(:head_kinleft, "root"), head_kinright = headcode(:head_kinright, "root"))
  lo, hi = minmax(i, h)''')
rep('''    head_sibleft = headcode(:head_sibleft, near([j for j in sibs if j < i])), head_sibright = headcode(:head_sibright, near([j for j in sibs if j > i])))''',
'''    head_sibleft = headcode(:head_sibleft, near([j for j in sibs if j < i])), head_sibright = headcode(:head_sibright, near([j for j in sibs if j > i])),
    head_kinleft = goldkin(i, -1, words), head_kinright = goldkin(i, 1, words))''')
rep('''"Gold head-property label codes of word i, whose head is word h (0: root)."''',
'''"Gold kinship code of word i (not the root) to its neighbour on side dir in the sentence `words`."
goldkin(i, dir, words) = 1 <= i + dir <= length(words) ?
  kin_between(u -> words[u].head == 0 ? NONE : UInt32(words[u].head), i, i + dir, UInt32(words[i].head), length(words)) : KIN_NONE

"Gold head-property label codes of word i, whose head is word h (0: root)."''')
# consistency + value
rep('''  elseif name == :head_wc300 || name == :head_wc30 || haskey(HEADSIB_KIND, name)
    (hv = head_value(st, t, h, name)) == PENDING || hv == v''', '''  elseif name == :head_wc300 || name == :head_wc30 || haskey(HEADSIB_KIND, name) || haskey(HEADKIN_DIR, name)
    (hv = head_value(st, t, h, name)) == PENDING || hv == v''')
rep('''  elseif name == :head_wc300 || name == :head_wc30
    wclass_gold(''', '''  elseif haskey(HEADKIN_DIR, name)                   # t's kinship to that neighbour under h, or pending
    kin_value(st, t, h, HEADKIN_DIR[name])
  elseif name == :head_wc300 || name == :head_wc30
    wclass_gold(''')
# default_specs
rep('''headsibs::Vector{Symbol} = Symbol[], idxeq::Bool = false)
  all(n -> haskey(HEADSIB_KIND, n), headsibs)''', '''headsibs::Vector{Symbol} = Symbol[], idxeq::Bool = false,
  kinitems::Bool = false, headkin::Vector{Symbol} = Symbol[])
  all(n -> haskey(HEADKIN_DIR, n), headkin) || error("headkin: names are $(join(keys(HEADKIN_DIR), ", "))")
  all(n -> haskey(HEADSIB_KIND, n), headsibs)''')
rep('''          idxeq ? idxeq_items(st, t, c) : UInt64[],
''', '''          idxeq ? idxeq_items(st, t, c) : UInt64[],
          kinitems ? kin_items(st, t, c) : UInt64[],
''')
rep('''hwnames; headsibs]],''', '''hwnames; headsibs; headkin]],''')
rep('''      [FeatPath(n) for n in headsibs]),''', '''      [FeatPath(n) for n in headsibs],
      [FeatPath(n) for n in headkin]),''')
rep('''length(hwnames) + length(headsibs) || error''', '''length(hwnames) + length(headsibs) + length(headkin) || error''')
open(p, "w").write(s); print("ok")
