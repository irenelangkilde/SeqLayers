# specs.jl  --  default layers: chars (tokenization) -> tokens (parsing) -> sentences
#
# Customize the model by editing these specs (features, relations, context paths,
# decisions, constraints) or by building your own Vector{LayerSpec}.

# Gap after a char: I = token continues, B = token ends (no space), S = token ends (space follows),
# E = token and sentence end.  Sentences can be closed from either side: committing E closes the
# tokens before it, and a closed tree in the token layer sets E after its last token.
const GAP_I = UInt32(1)
const GAP_B = UInt32(2)
const GAP_S = UInt32(3)
const GAP_E = UInt32(4)
const GAP_NAMES = ["I", "B", "S", "E"]

const DEPRELS = ["acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf",
  "compound", "conj", "cop", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed",
  "flat", "goeswith", "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan",
  "parataxis", "punct", "reparandum", "root", "vocative", "xcomp"]
const DEPREL_CODES = UInt32.(1:length(DEPRELS))
deprelcode(s::AbstractString) =
  UInt32(something(findfirst(==(first(split(s, ':'))), DEPRELS), findfirst(==("dep"), DEPRELS)))

# Token tag decisions.  Each is decided easy-first on its own, so any of them can inform the
# others and the attachment before the rest are known.  Gold: upos and xpos from CoNLL-U columns
# 4 and 5; the others from the PTB tag and the form, as in word-embeddings_3's defsOfFeatFs.jl.
# The last label of each set is the fallback (unknown tag, misaligned token).
const TAGSETS = (
  upos = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
    "PUNCT", "SCONJ", "SYM", "VERB", "X"],
  xpos = ["NN", "IN", "DT", "PRP", ".", "RB", "VB", "JJ", "NNP", "VBP", "NNS", ",", "VBZ", "CC", "VBD",
    "VBG", "TO", "CD", "MD", "PRP\$", "VBN", "UH", "?", "WP", "WRB", "RP", "WDT", ":", "-RRB-", "POS",
    "-LRB-", "``", "''", "HYPH", "JJR", "NNPS", "EX", "RBR", "JJS", "PDT", "FW", "SYM", "!", "NFP",
    "ADD", "\$", "GW", "RBS", "LS", "WP\$", "AFX", "\"", "XX"],
  basepos = ["J", "N", "R", "V", "0"],
  number = ["singular", "plural", "0"],
  person = ["1st", "2nd", "3rd", "0"],
  tense = ["present", "past", "infinitive", "0"])
tagcode(name::Symbol, s::AbstractString) =
  UInt32(something(findfirst(==(s), TAGSETS[name]), length(TAGSETS[name])))
const TAG_CODES = map(ls -> UInt32.(1:length(ls)), TAGSETS)
const TAG_ITEMTAGS = NamedTuple{keys(TAGSETS)}(map(n -> hash("token." * String(n)), keys(TAGSETS)))

const FIRST_PERSON = Set(["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"])
const SECOND_PERSON = Set(["you", "your", "yours", "yourself", "yourselves"])
isnounpos(p) = startswith(p, "N") && p != "NFP"      # NFP (superfluous punctuation) isn't a noun

"Gold label codes for every tag decision of one word."
function gold_tags(form::AbstractString, upos::AbstractString, xpos::AbstractString)
  f = lowercase(form)
  basepos = startswith(xpos, "J") ? "J" : isnounpos(xpos) || startswith(xpos, "CD") ? "N" :
    startswith(xpos, "RB") ? "R" : startswith(xpos, "V") ? "V" : "0"
  number = xpos == "CD" ? (f in ("1", "one") ? "singular" : "plural") :
    xpos in ("NN", "NNP", "VBZ") ? "singular" : xpos in ("NNS", "NNPS", "VBP") ? "plural" : "0"
  # extended from nouns only to personal and possessive pronouns, so 1st/2nd person can occur
  person = !(isnounpos(xpos) || xpos in ("PRP", "PRP\$")) ? "0" :
    f in FIRST_PERSON ? "1st" : f in SECOND_PERSON ? "2nd" : "3rd"
  tense = xpos == "VBZ" ? "present" : xpos == "VBD" ? "past" : xpos == "VB" ? "infinitive" : "0"
  (upos = tagcode(:upos, upos), xpos = tagcode(:xpos, xpos), basepos = tagcode(:basepos, basepos),
    number = tagcode(:number, number), person = tagcode(:person, person), tense = tagcode(:tense, tense))
end

# Head-property decisions (roadmap step 3): properties of a token's head, decided easy-first like
# any other decision, usually before the head itself is.  Once committed they narrow the
# attachment candidates (strictly: a wrong one can exclude the true head), and a token left with
# a single candidate gets that attachment (`infer_attach!`).  Gold from the gold tree; distance
# is in words.  The last label of each set is the fallback.
const HEADSETS = (
  head_upos = [filter(!=("X"), TAGSETS.upos); "ROOT"; "X"],
  head_xpos = [filter(!=("XX"), TAGSETS.xpos); "ROOT"; "XX"],
  head_dir = ["left", "right", "root"],
  head_dist = ["1", "2", "3-5", "6-10", "11+", "root"])
# Optional head properties (default_specs(headsdist = true)): the head's signed distance, direction
# and distance in one label (L = head to the left).
# head_trank (default_specs(headtrank = true)): the token's tree rank under its head -- its rank among the head's
# dependents on its side counting outward (1 = nearest; see tree_rank).
const ALLHEADSETS = merge(HEADSETS, (
  head_sdist = ["L11+", "L6-10", "L3-5", "L2", "L1", "R1", "R2", "R3-5", "R6-10", "R11+", "root"],
  head_trank = ["1", "2", "3", "4+", "root"]))
tranklabel(r::Int) = r >= 4 ? "4+" : string(r)
headcode(name::Symbol, s::AbstractString) =
  UInt32(something(findfirst(==(s), ALLHEADSETS[name]), length(ALLHEADSETS[name])))
const HEAD_CODES = map(ls -> UInt32.(1:length(ls)), ALLHEADSETS)
const HEAD_ITEMTAGS = NamedTuple{keys(ALLHEADSETS)}(map(n -> hash("token." * String(n)), keys(ALLHEADSETS)))
distlabel(d::Int) = d == 1 ? "1" : d == 2 ? "2" : d <= 5 ? "3-5" : d <= 10 ? "6-10" : "11+"
isrootlabel(name::Symbol, v::UInt32) = lowercase(ALLHEADSETS[name][v]) == "root"
"The head-property decisions a token layer has (the standard four, plus any optional ones)."
headnames(toks) = [d.name for d in toks.spec.decisions if haskey(ALLHEADSETS, d.name)]

"Gold head-property label codes of word i, whose head is word h (0: root)."
function gold_head(i::Int, h::Int, words)
  h == 0 && return (head_upos = headcode(:head_upos, "ROOT"), head_xpos = headcode(:head_xpos, "ROOT"),
    head_dir = headcode(:head_dir, "root"), head_dist = headcode(:head_dist, "root"), head_sdist = headcode(:head_sdist, "root"),
    head_trank = headcode(:head_trank, "root"))
  lo, hi = minmax(i, h)
  rank = 1 + count(j -> j != i && words[j].head == h, lo+1:hi-1)   # h's other dependents between h and i
  (head_upos = headcode(:head_upos, words[h].upos), head_xpos = headcode(:head_xpos, words[h].xpos),
    head_dir = headcode(:head_dir, h < i ? "left" : "right"), head_dist = headcode(:head_dist, distlabel(abs(h - i))),
    head_sdist = headcode(:head_sdist, (h < i ? "L" : "R") * distlabel(abs(h - i))), head_trank = headcode(:head_trank, tranklabel(rank)))
end

const MAX_LINK_CHARS = 200     # head candidates must start within this many chars
"Each token layer spec's attachment link paths (set by default_specs; read by generalization_slots)."
const LINKPATHS = IdDict{Any,Vector{Int}}()
# token context paths used in attachment features: form, caps, len, neighbors' forms (1:5),
# the six tags (7:12) and the neighbors' upos/xpos (13:16); not the head's (6, 17, 18).  With word
# classes, their paths (23:26 for four levels) are added, then any suffix paths.
const LINK_PATHS = [1:5; 7:16]

charclass(c::Char) =
  isspace(c) ? 1 : islowercase(c) ? 2 : isuppercase(c) ? 3 : isdigit(c) ? 4 : ispunct(c) ? 5 : 6
capsclass(s::AbstractString) =
  all(isuppercase, s) ? 1 : isuppercase(first(s)) ? 2 : any(isuppercase, s) ? 3 : 4

segchar(st, i) = st.doc.text[st.span.start+i-1]
nchars(st) = length(st.span)
docpos(st, i) = st.span.start + i - 1

##======== char layer

function gap_candidates(st, i)
  isspace(segchar(st, i)) && return UInt32[]
  st.final && i == nchars(st) && return [GAP_E]                 # end of document
  (i == nchars(st) || isspace(segchar(st, i + 1))) || return [GAP_I, GAP_B]
  arc_crosses(st, i) ? [GAP_S] : [GAP_S, GAP_E]                  # no sentence end under a committed arc
end

"Gap candidates before any constraint (no check for arcs across the gap): what the model must tell apart."
function gap_allcandidates(st, i)
  isspace(segchar(st, i)) && return UInt32[]
  st.final && i == nchars(st) && return [GAP_E]
  (i == nchars(st) || isspace(segchar(st, i + 1))) ? [GAP_S, GAP_E] : [GAP_I, GAP_B]
end

"""
Mask for recording gap i from the gold state: hide the gap and the tokens touching it -- they
exist, and have their forms, only because of this gap -- so paths into them read as PENDING.
"""
function gap_mask(st, key)
  i = key[3]
  chars = st.layers[:char]
  old = chars[:gap, i]
  chars[:gap, i] = PENDING
  hid = Tuple{Symbol,Int}[]
  for j in (i, i + 1)
    j <= length(chars) || continue
    t = chars[:token_of, j]
    (t == NONE || t == PENDING) && continue
    (:token, Int(t)) in st.hidden || (push!(st.hidden, (:token, Int(t))); push!(hid, (:token, Int(t))))
  end
  () -> (chars[:gap, i] = old; foreach(h -> delete!(st.hidden, h), hid))
end

"Does a committed attachment link a token ending at or before char j to one starting after it?"
function arc_crosses(st, j)
  toks = st.layers[:token]
  for t in 1:length(toks)
    h = toks[:attach, t]
    (h == NONE || h == PENDING) && continue
    l, r = toks[:first_char, t] < toks[:first_char, h] ? (t, Int(h)) : (Int(h), t)
    toks[:last_char, l] <= j < toks[:first_char, r] && return true
  end
  false
end

"The token span containing char j, if every gap delimiting it has been decided."
function completed_span(st, j)
  chars = st.layers[:char]
  a = j
  while a > 1 && !isspace(segchar(st, a - 1))
    g = chars[:gap, a-1]
    g == PENDING && return nothing
    g == GAP_I || break
    a -= 1
  end
  b = j
  while true
    g = chars[:gap, b]
    g == PENDING && return nothing
    g == GAP_I || break
    b += 1
  end
  a:b
end

"Tokenization constraint: once a span's gaps are all decided, create its token."
function tokenize!(st, i)
  chars = st.layers[:char]
  for j in (i, i + 1)
    1 <= j <= length(chars) || continue
    chars[:token_of, j] == PENDING || continue    # whitespace, or already tokenized
    span = completed_span(st, j)
    span === nothing && continue
    t = addcol!(st, :token, :first_char => first(span), :last_char => last(span))
    for k in span
      chars[:token_of, k] = t
    end
  end
end

##======== token layer

tokstr(st, t) = String(st.doc.text[docpos(st, Int(st.layers[:token][:first_char, t])):docpos(st, Int(st.layers[:token][:last_char, t]))])
tokspan(st, t) = docpos(st, Int(st.layers[:token][:first_char, t])):docpos(st, Int(st.layers[:token][:last_char, t]))

"The token next to t in direction dir (-1/+1): NONE at the segment edge, PENDING if not tokenized yet."
function adjacent_token(st, t, dir)
  toks, chars = st.layers[:token], st.layers[:char]
  j = Int(dir < 0 ? toks[:first_char, t] : toks[:last_char, t]) + dir
  while 1 <= j <= length(chars) && isspace(segchar(st, j))
    j += dir
  end
  1 <= j <= length(chars) || return NONE
  chars[:token_of, j]
end

function creates_cycle(st, t, h)
  toks = st.layers[:token]
  for _ in 1:length(toks)
    h == t && return true
    v = toks[:attach, h]
    (v == NONE || v == PENDING) && return false
    h = Int(v)
  end
  true
end

"Is there a committed E between tokens a and b?"
function crosses_E(st, a, b)
  toks, chars = st.layers[:token], st.layers[:char]
  l, r = toks[:first_char, a] < toks[:first_char, b] ? (a, b) : (b, a)
  any(j -> chars[:gap, j] == GAP_E, Int(toks[:last_char, l]):Int(toks[:first_char, r])-1)
end

"""
Would an arc between tokens t and h (NONE: the root, at position 0) cross a committed arc?  Arcs
are compared by the tokens' first chars; two arcs cross when exactly one end of one lies strictly
inside the other.
"""
function crosses_arc(st, t, h)
  toks = st.layers[:token]
  pos(u) = Int(toks[:first_char, u])
  a, b = h == NONE ? (0, pos(t)) : minmax(pos(t), pos(Int(h)))
  for u in 1:length(toks)
    u == t && continue
    g = toks[:attach, u]
    g == PENDING && continue
    c, d = g == NONE ? (0, pos(u)) : minmax(pos(u), pos(Int(g)))
    (a < c < b < d || c < a < d < b) && return true
  end
  false
end

"""
Attachment candidates for token t.  With `projective`, candidates whose arc would cross a committed
arc are dropped -- unless that leaves none, when they're kept (the gold tree isn't always projective).
"""
function attach_candidates(st, t; maxchars = MAX_LINK_CHARS, projective = false)
  cands = attach_candidates_(st, t; maxchars)
  projective || return cands
  proj = filter(h -> !crosses_arc(st, t, h), cands)
  isempty(proj) && !STRICT[] ? cands : proj            # (search: no fallback; an empty set is a failure)
end

function attach_candidates_(st, t; maxchars = MAX_LINK_CHARS)
  toks = st.layers[:token]
  s = toks[:sentence, t]
  # NONE = attach to root; a closed sentence gets exactly one root
  hasroot = s != PENDING && any(u -> u != t && toks[:sentence, u] == s && toks[:attach, u] == NONE, 1:length(toks))
  cands = hasroot ? UInt32[] : UInt32[NONE]
  ft = Int(toks[:first_char, t])
  for h in 1:length(toks)
    h == t && continue
    toks[:sentence, h] == s || continue           # same sentence, or both not yet in one
    abs(Int(toks[:first_char, h]) - ft) <= maxchars || continue
    creates_cycle(st, t, h) && continue
    crosses_E(st, t, h) && continue
    push!(cands, UInt32(h))
  end
  # In a closed sentence the token must attach somewhere: drop the distance limit if needed
  isempty(cands) && s != PENDING && maxchars < typemax(Int) && return attach_candidates_(st, t; maxchars = typemax(Int))
  filter!(h -> head_consistent(st, t, h), cands)
end

"""
Words from token t to token h (1 = adjacent), or nothing while some char between them isn't
tokenized yet.
"""
function word_distance(st, t, h)
  toks, chars = st.layers[:token], st.layers[:char]
  l, r = toks[:first_char, t] < toks[:first_char, h] ? (t, Int(h)) : (Int(h), t)
  n, prev = 0, UInt32(0)
  a, b = Int(toks[:last_char, l]), Int(toks[:first_char, r])
  (toks[:last_char, l] == PENDING || b - 1 > nchars(st)) && return nothing   # (a token not fully in this segment yet)
  for j in a+1:b-1
    isspace(segchar(st, j)) && continue
    u = chars[:token_of, j]
    u == PENDING && return nothing
    u != prev && (n += 1; prev = u)
  end
  n + 1
end

"Does head candidate h (NONE: root) agree with t's committed head-property decisions?  Undecided values agree."
head_consistent(st, t, h) = all(name -> head_consistent_one(st, t, h, name), headnames(st.layers[:token]))

"Does head candidate h agree with t's committed head property `name` (pending: yes)?"
function head_consistent_one(st, t, h, name)
  toks = st.layers[:token]
  v = toks[name, t]
  v == PENDING && return true
  isrootlabel(name, v) == (h == NONE) || return false
  h == NONE && return true
  if name == :head_sdist                            # direction and distance in one label
    dirc = toks[:first_char, h] < toks[:first_char, t] ? "L" : "R"
    lab = ALLHEADSETS.head_sdist[v]
    startswith(lab, dirc) || return false
    d = word_distance(st, t, h)
    d === nothing || lab == dirc * distlabel(d)
  elseif name == :head_upos || name == :head_xpos
    tag = toks[name == :head_upos ? :upos : :xpos, h]
    tag == PENDING || HEADSETS[name][v] == TAGSETS[name == :head_upos ? :upos : :xpos][tag]
  elseif name == :head_trank                         # the rank only grows as dependents attach between h and t
    want = v == headcode(:head_trank, "4+") ? 4 : parse(Int, ALLHEADSETS.head_trank[v])
    r = tree_rank(st, t, h)
    r >= want && return want == 4 || r == want
    lo, hi = minmax(toks[:first_char, t], toks[:first_char, h])
    any(u -> u != t && lo < toks[:first_char, u] < hi && toks[:attach, u] == PENDING, 1:length(toks))   # may still reach it
  elseif name == :head_dir
    HEADSETS.head_dir[v] == (toks[:first_char, h] < toks[:first_char, t] ? "left" : "right")
  else
    d = word_distance(st, t, h)
    d === nothing || HEADSETS.head_dist[v] == distlabel(d)
  end
end

"""
Forward checking (FWDCHECK, set for decoding): once no other head can appear for t, a value of head
property `name` is a candidate only if some possible head of t (attach_candidates: t's other committed head
properties, cycles, a root already taken, projectivity) has it, or its value isn't known yet.  Without
FWDCHECK, or while more heads may appear, every value.  Outside search, an empty set falls back to every value.
"""
const FWDCHECK = Ref(false)

"""
Soft head properties (set for decoding): head-property decisions are never committed by the model (only
derived from the committed attachment); instead they weight the attachment's candidates (decide: the
product of the ceiling's domains=product).
"""
const SOFTHEADS = Ref(false)

function head_candidates(st, t, name, projective)
  all_ = HEAD_CODES[name]
  (FWDCHECK[] && st.infer) || return all_
  more_heads_possible(st, t) && return all_
  heads = attach_candidates(st, t; projective)
  ok = falses(length(all_))
  for h in heads
    v = head_value(st, t, h, name)
    v == PENDING && return all_                         # some head's value isn't known yet: no pruning
    ok[v] = true
  end
  out = all_[ok]
  isempty(out) && !STRICT[] ? all_ : out
end

"Could another attachment candidate for t still appear?  Not once its sentence is closed, or every char in reach is tokenized."
function more_heads_possible(st, t)
  toks, chars = st.layers[:token], st.layers[:char]
  toks[:sentence, t] != PENDING && return false
  ft = Int(toks[:first_char, t])
  hi = ft + MAX_LINK_CHARS
  hi > nchars(st) && !st.final && return true       # reach extends into the next segment
  any(j -> !isspace(segchar(st, j)) && chars[:token_of, j] == PENDING, max(1, ft - MAX_LINK_CHARS):min(hi, nchars(st)))
end

"""
Selection constraint: a token whose committed head-property decisions leave exactly one
attachment candidate, when no other candidate can still appear, gets that attachment.  (A root
label excludes every token, so it never has to wait.)  Not used by the oracle, which commits gold.
"""
function infer_attach!(st, j)
  st.infer || return
  toks = st.layers[:token]
  k = findfirst(d -> d.name == :attach, toks.spec.decisions)
  for t in 1:length(toks)
    toks[:attach, t] == PENDING || continue
    decided = [name for name in headnames(toks) if toks[name, t] != PENDING]
    isempty(decided) && continue
    cands = toks.spec.decisions[k].candidates(st, t)       # the decision's own (e.g. projective) candidates
    length(cands) == 1 || continue
    h = only(cands)
    (h == NONE && any(name -> isrootlabel(name, toks[name, t]), decided)) || !more_heads_possible(st, t) || continue
    settle!(st, :token, k, t, h)
    refresh!(st)
  end
end

"Set a decision outside the agenda (inferred or derived), and take it off the agenda."
function settle!(st, lname, k, i, v)
  seq = st.layers[lname]
  seq[seq.spec.decisions[k].name, i] = v
  key = (lname, k, i)
  haskey(st.agenda, key) && delete!(st.agenda, key)
  delete!(st.best, key); delete!(st.reads, key)
  filter!(!=(key), st.fresh)
end

"""
The value of head property `name` that attaching t to h (NONE: root) gives: PENDING while it
isn't known yet (the words between not tokenized, or the head's tag not decided).
"""
function head_value(st, t, h, name)
  toks = st.layers[:token]
  if h == NONE
    ALLHEADSETS[name] |> ls -> UInt32(findfirst(l -> lowercase(l) == "root", ls))
  elseif name == :head_sdist
    d = word_distance(st, t, h)
    d === nothing ? PENDING : headcode(name, (toks[:first_char, h] < toks[:first_char, t] ? "L" : "R") * distlabel(d))
  elseif name == :head_trank
    headcode(name, tranklabel(tree_rank(st, t, h)))
  elseif name == :head_dir
    headcode(name, toks[:first_char, h] < toks[:first_char, t] ? "left" : "right")
  elseif name == :head_dist
    d = word_distance(st, t, h)
    d === nothing ? PENDING : headcode(name, distlabel(d))
  else
    tag = toks[name == :head_upos ? :upos : :xpos, h]
    tag == PENDING ? PENDING : headcode(name, TAGSETS[name == :head_upos ? :upos : :xpos][tag])
  end
end

"""
Backward step: a committed attachment determines its token's head properties.  Fills those still
pending as far as they're known: all of them for a root; direction always; distance once the
words between are tokenized; the head's tags once they're decided (so a later tag commit fills
them in -- meanwhile they aren't ready, so the model can't decide them).  Properties committed
before the attachment are left as they are.  Not used by the
oracle, which commits gold itself.
"""
function derive_heads!(st, j)
  st.infer || return
  toks = st.layers[:token]
  ks = Dict(d.name => k for (k, d) in enumerate(toks.spec.decisions) if haskey(ALLHEADSETS, d.name))
  changed = false
  for t in 1:length(toks)
    h = toks[:attach, t]
    h == PENDING && continue
    for (name, k) in ks
      toks[name, t] == PENDING || continue
      v = head_value(st, t, h, name)
      v == PENDING && continue
      settle!(st, :token, k, t, v)
      changed = true
    end
  end
  changed && refresh!(st)
end

"Mask for recording an attachment: hide it and t's head-property decisions, which reveal it."
function attach_mask(st, key)
  t = key[3]
  toks = st.layers[:token]
  names = [:attach; headnames(toks)]
  old = [toks[n, t] for n in names]
  foreach(n -> (toks[n, t] = PENDING), names)
  refresh!(st)
  () -> (foreach(((n, v),) -> (toks[n, t] = v), zip(names, old)); refresh!(st))
end

"Mask for recording a head-property decision: hide it and the attachment, which reveals it."
function head_mask(st, key)
  t = key[3]
  toks = st.layers[:token]
  name = toks.spec.decisions[key[2]].name
  old = (toks[name, t], toks[:attach, t])
  toks[name, t] = PENDING; toks[:attach, t] = PENDING
  refresh!(st)
  () -> (toks[name, t] = old[1]; toks[:attach, t] = old[2]; refresh!(st))
end

"Head candidates before any constraint: root and every other token within MAX_LINK_CHARS."
function attach_allcandidates(st, t)
  toks = st.layers[:token]
  ft = Int(toks[:first_char, t])
  UInt32[NONE; [UInt32(h) for h in 1:length(toks) if h != t && abs(Int(toks[:first_char, h]) - ft) <= MAX_LINK_CHARS]]
end

"""
Safety net before output: any attachment or relation still pending inside a closed sentence
(e.g. left blocked at the end of a segment) is settled -- attached to the sentence's root if
that makes no cycle, else made a root -- and gets relation "dep".
"""
function settle_closed!(st)
  toks = st.layers[:token]
  for t in 1:length(toks)
    s = toks[:sentence, t]
    s == PENDING && continue
    if toks[:attach, t] == PENDING
      r = findfirst(u -> toks[:sentence, u] == s && toks[:attach, u] == NONE, 1:length(toks))
      toks[:attach, t] = (r === nothing || creates_cycle(st, t, r)) ? NONE : r
    end
    toks[:deprel, t] == PENDING && (toks[:deprel, t] = deprelcode("dep"))
  end
end

function distbucket(d)
  a = abs(d)
  b = a == 0 ? 0 : a <= 3 ? 1 : a <= 8 ? 2 : a <= 20 ? 3 : a <= 50 ? 4 : 5
  sign(d) * b
end

const ATTACH_TAG = hash("token.attach")
const DEPREL_TAG = hash("token.deprel")
const GAP_TAG = hash("char.gap")

attach_dist(st, t, c) =
  c == NONE ? 0 : distbucket(Int(st.layers[:token][:first_char, c]) - Int(st.layers[:token][:first_char, t]))

attach_features(st, t, c, paths = LINK_PATHS) = link_features(st, :token, ATTACH_TAG, t, c, attach_dist(st, t, c), paths)
attach_items(st, t, c, paths = LINK_PATHS; rootabsent = false) =
  link_items(st, :token, ATTACH_TAG, t, c, attach_dist(st, t, c), paths; rootabsent)

##======== sibling and child relations (relations = true)
#
# For a token t with head h: its left/right sibling (the nearest token on that side with the same
# head), inner sibling (the nearest one between t and h) and outer sibling (the nearest one beyond t,
# on t's side of h); and its nearest left dependent and first and second nearest right dependents
# (dep_left1, dep_right1, dep_right2).  All read committed attachments, so they're absent (PENDING)
# while what they depend on is undecided -- including "no such child yet".  For attachment, the
# siblings t would have under each candidate head c (c's children placed relative to t) are items of
# the candidate, so they can inform the attachment itself.  Each relation's distance from t (word
# buckets as for heads) is a row of its own; inner and outer siblings also get their direction.

const SIB_KINDS = (:left_sibling, :right_sibling, :inner_sibling, :outer_sibling)
const CHILD_KINDS = (:dep_left1, :dep_right1, :dep_right2)
const REL_KINDS = (SIB_KINDS..., CHILD_KINDS...)
const DIR_KINDS = (:inner_sibling, :outer_sibling)          # the others have a fixed direction
const SIB_TAG = hash("token.attach.sibling")
const TRANK_TAG = hash("token.attach.treerank")

"""
Tree rank of t under head h: 1 + h's attached dependents (t excluded) between h and t, i.e. t would be h's
k-th dependent on its side counting outward (1 = nearest).  The root: 0.  Counts committed dependents only.
"""
function tree_rank(st, t, h)
  h == NONE && return 0
  h == PENDING && return -1
  toks = st.layers[:token]
  pt, ph = toks[:first_char, t], toks[:first_char, Int(h)]
  lo, hi = minmax(pt, ph)
  1 + count(u -> u != t && lo < toks[:first_char, u] < hi, children_index(st)[Int(h)])
end

"Tree-rank bucket code for attachment items: 1, 2, 3, 4+; root 5."
trank_code(r) = r < 0 ? PENDING : r == 0 ? UInt32(5) : UInt32(min(r, 4))

"""
Each token's committed children, by position: shared during a refresh! via st.memo, and between
refreshes via st.relcache while the attach row is unchanged.
"""
function children_index(st)
  m = st.memo
  m !== nothing && haskey(m, :children) && return m[:children]
  toks = st.layers[:token]
  att = UInt32[toks[:attach, u] for u in 1:toks.ncols]
  c = st.relcache
  c !== nothing && c[1] == att && (m !== nothing && (m[:children] = c[2]); return c[2])
  ch = [Int[] for _ in 1:toks.ncols]
  for u in 1:toks.ncols
    h = toks[:attach, u]
    (h == PENDING || h == NONE) && continue
    push!(ch[Int(h)], u)
  end
  foreach(v -> sort!(v; by = u -> toks[:first_char, u]), ch)
  m !== nothing && (m[:children] = ch)
  st.relcache = (att, ch)
  ch
end

"Sibling of t of the given kind among head h's children (t excluded), or PENDING."
function sibling_of(st, t, h, kind)
  (h == PENDING || h == NONE) && return PENDING
  toks = st.layers[:token]
  pos(u) = toks[:first_char, u]
  pt, ph = pos(t), pos(Int(h))
  sibs = [u for u in children_index(st)[Int(h)] if u != t]
  pick(f, near) = (c = filter(f, sibs); isempty(c) ? PENDING : UInt32(near(c)))
  kind == :left_sibling && return pick(u -> pos(u) < pt, last)
  kind == :right_sibling && return pick(u -> pos(u) > pt, first)
  if kind == :inner_sibling
    return ph > pt ? pick(u -> pt < pos(u) < ph, first) : pick(u -> ph < pos(u) < pt, last)
  end
  ph > pt ? pick(u -> pos(u) < pt, last) : pick(u -> pos(u) > pt, first)      # :outer_sibling
end

"Child of t of the given kind (dep_left1, dep_right1, dep_right2), or PENDING."
function child_of(st, t, kind)
  toks = st.layers[:token]
  pt = toks[:first_char, t]
  ch = children_index(st)[t]
  if kind == :dep_left1
    l = filter(u -> toks[:first_char, u] < pt, ch)
    return isempty(l) ? PENDING : UInt32(last(l))
  end
  r = filter(u -> toks[:first_char, u] > pt, ch)
  k = kind == :dep_right1 ? 1 : 2
  length(r) >= k ? UInt32(r[k]) : PENDING
end

"""
Relation variants (default_specs relbounded / relnone).  `bounded`: left and right siblings don't
cross the head (a token whose head is on its left has as left sibling only a sibling between the head
and itself; likewise on the right).  `none`: a relation that certainly doesn't exist -- every token
where it could be is attached (in a closed sentence) and none is it -- is NONE, a value of its own,
while one that may still appear stays PENDING (absent).
"""
struct RelFlags
  bounded::Bool
  none::Bool
end
const RELPLAIN = RelFlags(false, false)

"""
Tokens by position for range checks during a refresh! (st.memo): each token's rank, its sentence's
first and last rank, and running counts of unattached tokens by rank.
"""
function relranks(st)
  build() = begin
    toks = st.layers[:token]
    n = toks.ncols
    ord = sortperm([toks[:first_char, u] for u in 1:n])
    rank = zeros(Int, n)
    foreach(((r, u),) -> (rank[u] = r), enumerate(ord))
    cp = zeros(Int, n + 1)                              # cp[r+1]: unattached tokens among ranks 1..r
    for (r, u) in enumerate(ord)
      cp[r+1] = cp[r] + (toks[:attach, u] == PENDING)
    end
    sfirst, slast = zeros(Int, n), zeros(Int, n)        # by rank: the rank range of the token's sentence
    r = 1
    while r <= n
      sid = toks[:sentence, ord[r]]
      q = r
      while q < n && toks[:sentence, ord[q+1]] == sid
        q += 1
      end
      sfirst[r:q] .= r; slast[r:q] .= q
      r = q + 1
    end
    (ord, rank, cp, sfirst, slast)
  end
  m = st.memo
  m === nothing && return build()
  get!(build, m, :relranks)
end

"""
Is every token strictly between ranks lo and hi attached, with t in a closed sentence (so nothing more
can appear there)?  lo / hi: ranks (0 and n + 1 for the sentence's edges are clamped to it).
"""
function rel_range_done(st, rk, t, lo, hi)
  st.layers[:token][:sentence, t] == PENDING && return false
  ord, rank, cp, sfirst, slast = rk
  r = rank[t]
  lo = max(lo, sfirst[r] - 1); hi = min(hi, slast[r] + 1)
  hi - lo <= 1 && return true
  cp[hi] - cp[lo+1] == 0
end

"""
t's sibling of the given kind under head h (t excluded), under flags f: the token, NONE (certainly
none: f.none), or PENDING.
"""
function sibling_value(st, t, h, kind, f::RelFlags)
  h == PENDING && return PENDING
  toks = st.layers[:token]
  rk = relranks(st)
  if h == NONE                                         # a root has no siblings (one root per sentence)
    return f.none && toks[:sentence, t] != PENDING ? NONE : PENDING
  end
  ord, rank, _, sfirst, slast = rk
  rt, rh = rank[t], rank[Int(h)]
  lo, hi = if kind == :left_sibling
    (f.bounded && rh < rt ? rh : 0, rt)
  elseif kind == :right_sibling
    (rt, f.bounded && rh > rt ? rh : length(rank) + 1)
  elseif kind == :inner_sibling
    rh < rt ? (rh, rt) : (rt, rh)
  else                                                 # :outer_sibling: beyond t, away from the head
    rh < rt ? (rt, length(rank) + 1) : (0, rt)
  end
  lo = max(lo, sfirst[rt] - 1); hi = min(hi, slast[rt] + 1)
  sibs = [u for u in children_index(st)[Int(h)] if u != t && lo < rank[u] < hi]
  if !isempty(sibs)                                    # the nearest to t
    return UInt32(rank[sibs[1]] > rt ? sibs[argmin([rank[u] for u in sibs])] : sibs[argmax([rank[u] for u in sibs])])
  end
  f.none && rel_range_done(st, rk, t, lo, hi) ? NONE : PENDING
end

"t's child of the given kind (dep_left1, dep_right1, dep_right2) under flags f: the token, NONE, or PENDING."
function child_value(st, t, kind, f::RelFlags)
  u = child_of(st, t, kind)
  (u != PENDING || !f.none) && return u
  rk = relranks(st)
  rt = rk[2][t]
  kind == :dep_left1 ? (rel_range_done(st, rk, t, 0, rt) ? NONE : PENDING) :
    (rel_range_done(st, rk, t, rt, length(rk[2]) + 1) ? NONE : PENDING)
end

"""
Every token's relations at once, from the children lists sorted by position (during a refresh!,
shared via st.memo): a sibling is a neighbour in its head's list, a child an entry in its own.
With flags other than RELPLAIN, from sibling_value / child_value.
"""
function relation_table(st, f::RelFlags)
  f == RELPLAIN && return relation_table(st)
  m = st.memo
  key = f.bounded ? (f.none ? :reltable_bn : :reltable_b) : :reltable_n
  haskey(m, key) && return m[key]
  toks = st.layers[:token]
  n = toks.ncols
  tab = Dict{Symbol,Vector{UInt32}}()
  for kind in SIB_KINDS
    tab[kind] = [sibling_value(st, t, toks[:attach, t], kind, f) for t in 1:n]
  end
  for kind in CHILD_KINDS
    tab[kind] = [child_value(st, t, kind, f) for t in 1:n]
  end
  for kind in REL_KINDS
    rel = tab[kind]
    tab[Symbol(kind, :_dist)] = [relation_dist(st, t, rel[t]) for t in 1:n]
    kind in DIR_KINDS && (tab[Symbol(kind, :_dir)] = [relation_dir(st, t, rel[t]) for t in 1:n])
  end
  m[key] = tab
end

function relation_table(st)
  m = st.memo
  haskey(m, :reltable) && return m[:reltable]
  toks = st.layers[:token]
  n = toks.ncols
  pos(u) = toks[:first_char, u]
  tab = Dict(k => fill(PENDING, n) for k in REL_KINDS)
  ch = children_index(st)
  for h in 1:n
    c = ch[h]
    ph = pos(h)
    for (i, t) in enumerate(c)
      pt = pos(t)
      i > 1 && (tab[:left_sibling][t] = UInt32(c[i-1]))
      i < length(c) && (tab[:right_sibling][t] = UInt32(c[i+1]))
      if ph > pt                                       # head to the right: inner toward it, outer to the left
        i < length(c) && pos(c[i+1]) < ph && (tab[:inner_sibling][t] = UInt32(c[i+1]))
        i > 1 && (tab[:outer_sibling][t] = UInt32(c[i-1]))
      else                                             # head to the left
        i > 1 && pos(c[i-1]) > ph && (tab[:inner_sibling][t] = UInt32(c[i-1]))
        i < length(c) && (tab[:outer_sibling][t] = UInt32(c[i+1]))
      end
    end
    # h's own children: nearest left, first and second nearest right
    k = searchsortedfirst(c, h; by = u -> pos(u))      # first child to the right of h
    k > 1 && (tab[:dep_left1][h] = UInt32(c[k-1]))
    k <= length(c) && (tab[:dep_right1][h] = UInt32(c[k]))
    k + 1 <= length(c) && (tab[:dep_right2][h] = UInt32(c[k+1]))
  end
  for kind in REL_KINDS                                # the distance (and direction) rows, from the same table
    rel = tab[kind]
    tab[Symbol(kind, :_dist)] = [relation_dist(st, t, rel[t]) for t in 1:n]
    kind in DIR_KINDS && (tab[Symbol(kind, :_dir)] = [relation_dir(st, t, rel[t]) for t in 1:n])
  end
  m[:reltable] = tab
end

relation_of(st, t, kind) = st.memo !== nothing ? relation_table(st)[kind][t] :
  kind in SIB_KINDS ? sibling_of(st, t, st.layers[:token][:attach, t], kind) : child_of(st, t, kind)
relation_of(st, t, kind, f::RelFlags) = f == RELPLAIN ? relation_of(st, t, kind) :
  st.memo !== nothing ? relation_table(st, f)[kind][t] :
  kind in SIB_KINDS ? sibling_value(st, t, st.layers[:token][:attach, t], kind, f) : child_value(st, t, kind, f)

"""
Word distance from t to u as `word_distance` computes it, in constant time during a refresh!: the
tokens' order by position and a running count of untokenized chars are built once per refresh
(st.memo).  Outside a refresh it falls back to word_distance.
"""
function word_distance_fast(st, t, u)
  m = st.memo
  m === nothing && return word_distance(st, t, u)
  idx = get!(m, :wordrank) do
    toks, chars = st.layers[:token], st.layers[:char]
    order = sortperm([toks[:first_char, v] for v in 1:toks.ncols])
    rank = zeros(Int, toks.ncols)
    foreach(((r, v),) -> (rank[v] = r), enumerate(order))
    pend = zeros(Int, chars.ncols + 1)                 # pend[j+1]: untokenized non-space chars in 1..j
    for j in 1:chars.ncols
      pend[j+1] = pend[j] + (!isspace(segchar(st, j)) && chars[:token_of, j] == PENDING)
    end
    (rank, pend)
  end
  rank, pend = idx
  toks = st.layers[:token]
  l, r = toks[:first_char, t] < toks[:first_char, u] ? (t, Int(u)) : (Int(u), t)
  a, b = Int(toks[:last_char, l]) + 1, Int(toks[:first_char, r]) - 1
  a <= b && pend[b+1] - pend[a] > 0 && return nothing   # something between isn't tokenized yet
  abs(rank[r] - rank[l])
end

"Word-distance bucket code (1: 1, 2: 2, 3: 3-5, 4: 6-10, 5: 11+) from t to u, or PENDING."
function relation_dist(st, t, u)
  (u == PENDING || u == NONE) && return u
  d = word_distance_fast(st, t, Int(u))
  d === nothing ? PENDING : UInt32(d == 1 ? 1 : d == 2 ? 2 : d <= 5 ? 3 : d <= 10 ? 4 : 5)
end

relation_dir(st, t, u) = u == PENDING || u == NONE ? u :
  UInt32(st.layers[:token][:first_char, Int(u)] < st.layers[:token][:first_char, t] ? 1 : 2)

"""
Items for the siblings t would have under candidate head c: for each kind, its upos, deprel and
distance from t (and the direction for inner and outer siblings).
"""
function hypothetical_sibling_items(st, t, c, f::RelFlags = RELPLAIN)
  toks = st.layers[:token]
  items = UInt64[]
  j = 0
  for kind in SIB_KINDS
    u = f == RELPLAIN ? (c == NONE ? PENDING : sibling_of(st, t, c, kind)) : sibling_value(st, t, c, kind, f)
    nv = u == PENDING || u == NONE                     # (no sibling token: its upos and deprel are the same value)
    vals = UInt32[nv ? u : toks[:upos, Int(u)], nv ? u : toks[:deprel, Int(u)], relation_dist(st, t, u)]
    kind in DIR_KINDS && push!(vals, relation_dir(st, t, u))
    for v in vals
      j += 1
      push!(items, item(v, (SIB_TAG, j, v)))
    end
  end
  items
end

"The relation rows (dynamic) and their context paths; the link paths among them (children only)."
function relation_rows(f::RelFlags = RELPLAIN)
  rels = RelDef[]
  for kind in REL_KINDS
    push!(rels, RelDef(kind, :token, true, (st, t) -> relation_of(st, t, kind, f)))
  end
  # distance and direction rows hold bucket codes, not columns: they're only ever read, never followed
  for kind in REL_KINDS
    dk = Symbol(kind, :_dist)
    push!(rels, RelDef(dk, :token, true, (st, t) -> st.memo !== nothing ? relation_table(st, f)[dk][t] :
      relation_dist(st, t, relation_of(st, t, kind, f))))
  end
  for kind in DIR_KINDS
    rk = Symbol(kind, :_dir)
    push!(rels, RelDef(rk, :token, true, (st, t) -> st.memo !== nothing ? relation_table(st, f)[rk][t] :
      relation_dir(st, t, relation_of(st, t, kind, f))))
  end
  paths = FeatPath[]
  link = Int[]                       # positions (within these paths) of the children's paths
  for kind in REL_KINDS
    for p in (FeatPath([kind], :upos), FeatPath([kind], :deprel), FeatPath(Symbol(kind, :_dist)))
      push!(paths, p)
      kind in CHILD_KINDS && push!(link, length(paths))
    end
  end
  foreach(k -> push!(paths, FeatPath(Symbol(k, :_dir))), DIR_KINDS)
  rels, paths, link
end

##======== sentence constraint (token layer)

"Is every non-space char strictly between base columns a and b tokenized?"
resolved(st, a, b) = all(j -> st.layers[:char][:token_of, j] != PENDING, a+1:b-1)

"Root token of t's attachment chain, or PENDING if the chain isn't finished."
function chain_root(st, t)
  toks = st.layers[:token]
  for _ in 1:length(toks)
    h = toks[:attach, t]
    h == PENDING && return PENDING
    h == NONE && return UInt32(t)
    t = Int(h)
  end
  PENDING
end

"All of the prefix is attached, inside itself, under exactly one root, and nothing outside attaches into it."
function isclosed(st, open, inprefix)
  toks = st.layers[:token]
  roots = 0
  for t in open
    h = toks[:attach, t]
    if inprefix[t]
      h == PENDING && return false
      h == NONE ? (roots += 1) : (inprefix[h] || return false)
    elseif h != PENDING && h != NONE && inprefix[h]
      return false
    end
  end
  roots == 1
end

function close_sentence!(st, ts)
  toks = st.layers[:token]
  s = addcol!(st, :sentence, :first_token => first(ts), :last_token => last(ts))
  for t in ts
    toks[:sentence, t] = s
  end
  st.closed_through = max(st.closed_through, Int(toks[:last_char, last(ts)]))
end

"Parse-side closure sets the char side to match: the gap after the sentence becomes E."
function mark_E!(st, j)
  chars = st.layers[:char]
  chars[:gap, j] = GAP_E
  key = (:char, 1, j)
  haskey(st.agenda, key) && delete!(st.agenda, key)
end

"""
Close the earliest sentence if possible, from either side:
 - char side: a committed E, once every char before it is tokenized, closes the tokens before it
   (their remaining attachments are then confined to the sentence);
 - parse side: a contiguous run of tokens forming one closed tree, followed by a token whose own
   tree has a different root (or by the end of the document), closes and sets E after it.
"""
function close_next_sentence!(st)
  toks, chars = st.layers[:token], st.layers[:char]
  open = sort!([t for t in 1:length(toks) if toks[:sentence, t] == PENDING]; by = t -> toks[:first_char, t])
  isempty(open) && return false
  e = findfirst(j -> chars[:gap, j] == GAP_E, st.closed_through+1:nchars(st))
  if e !== nothing
    j = st.closed_through + e
    ts = filter(t -> toks[:last_char, t] <= j, open)
    if !isempty(ts) && resolved(st, st.closed_through, j + 1)
      close_sentence!(st, ts)
      return true
    end
  end
  resolved(st, st.closed_through, Int(toks[:first_char, open[1]])) || return false
  inprefix = falses(length(toks))
  for (k, t) in enumerate(open)
    k > 1 && !resolved(st, Int(toks[:last_char, open[k-1]]), Int(toks[:first_char, t])) && return false
    inprefix[t] = true
    isclosed(st, open, inprefix) || continue
    if k == length(open)
      (st.final && resolved(st, Int(toks[:last_char, t]), nchars(st) + 1)) || return false
    else
      nxt = open[k+1]
      resolved(st, Int(toks[:last_char, t]), Int(toks[:first_char, nxt])) || return false
      r = chain_root(st, nxt)
      (r == PENDING || inprefix[r]) && continue
    end
    close_sentence!(st, open[1:k])
    mark_E!(st, Int(toks[:last_char, t]))
    return true
  end
  false
end

close_sentences!(st, _) = while close_next_sentence!(st) end

"Last resort at the end of a document (or a segment that never closes a sentence): lump the rest."
function force_close!(st)
  toks = st.layers[:token]
  open = sort!([t for t in 1:length(toks) if toks[:sentence, t] == PENDING]; by = t -> toks[:first_char, t])
  for t in 1:length(toks)
    toks[:attach, t] == PENDING && (toks[:attach, t] = NONE)
    toks[:deprel, t] == PENDING && (toks[:deprel, t] = deprelcode("dep"))
  end
  if isempty(open)
    st.closed_through = nchars(st)
  else
    close_sentence!(st, open)
    mark_E!(st, Int(toks[:last_char, last(open)]))
  end
  st.final && (st.closed_through = nchars(st))
end

##======== gold choices for the default layers (used by OracleModel)

function oracle_choice(st, key)
  gold = st.doc.gold
  lname, k, i = key
  if lname == :char
    g = gold.gap[docpos(st, i)]
    cands = gap_candidates(st, i)
    g in cands && return g
    g == GAP_E && GAP_B in cands && return GAP_B     # sentence end with no space after it
    return first(cands)
  end
  span = tokspan(st, i)
  name = st.layers[lname].spec.decisions[k].name
  iswclass(name) && return first(st.layers[lname].spec.decisions[k].candidates(st, i))   # gold first
  if haskey(TAGSETS, name) || haskey(ALLHEADSETS, name)
    fallback = UInt32(length(haskey(TAGSETS, name) ? TAGSETS[name] : ALLHEADSETS[name]))
    return haskey(gold.tags, name) ? get(gold.tags[name], span, fallback) : fallback
  end
  haskey(gold.head, span) || return k == 1 ? NONE : deprelcode("dep")    # misaligned token
  k == 2 && return gold.deprel[span]
  hs = gold.head[span]
  isempty(hs) && return NONE
  c = first(hs) - st.span.start + 1
  1 <= c <= nchars(st) || return PENDING          # head lies beyond this segment
  h = st.layers[:char][:token_of, c]
  (h == PENDING || h == NONE || tokspan(st, Int(h)) != hs) && return PENDING
  h
end

##======== the layer specs

"Suffix-array features for the gap after char i: label distribution code and context length reached."
function sa_feats(idx::CharIndex, side::Symbol)
  lastq = Ref{Any}(nothing)        # (doc, pos) => result of the latest query; dist and len ask in turn
  function query(st, i)
    k = (st.doc.text, docpos(st, i))
    (lastq[] !== nothing && lastq[][1][1] === k[1] && lastq[][1][2] == k[2]) || (lastq[] = (k, context_probs(idx, st.doc, k[2], side)))
    lastq[][2]
  end
  dist(st, i) = isspace(segchar(st, i)) ? NONE : distcode(first(query(st, i)))
  len(st, i) = isspace(segchar(st, i)) ? NONE : UInt32(1 + last(query(st, i)))
  [FeatDef(Symbol(:sa_, side), dist), FeatDef(Symbol(:sa_, side, :_len), len)]
end

"""
The default char -> token -> sentence layers.  With a `charindex` (built from training docs
with `CharIndex(docs)`), the char layer also gets long-context features :sa_left and :sa_right
(and :sa_left_len/:sa_right_len rows, which aren't in the context by default).  With
`wordclasses` (`load_wordclasses(dir)`), the token layer also gets a word-class decision per level
(see wordclass.jl), in the context and in the attachment features.  With `suffixes` (e.g. [3]),
it gets a feature per length n, the form's last n letters (:suffix3; the whole form if shorter),
also in the context and the attachment features (after the classes).  With `noise` = k > 0, a
feature :noise with a pseudo-random value 1..k per token occurrence (irrelevant by construction, for
testing that irrelevant features are discarded), likewise, last.  With `rootabsent`, the root
candidate's candidate-side attachment items are absent (see link_items).  With `projective`,
attachment candidates whose arc would cross a committed arc are dropped (see attach_candidates);
recorded events are unaffected (they use all candidates before constraints).  With `relations`,
the token layer gets sibling and child relations (see "sibling and child relations"): their upos,
deprel and distance (and direction) in the context, the children's also in the attachment
features, and each candidate's hypothetical siblings of the dependent as attachment items.  With
`headsdist`, a fifth head-property decision :head_sdist (signed distance: direction and distance in
one label), narrowing and derived like the others.  With `pptable` (`load_pptable`), attachment
records get two items from a large parsed corpus: how readily the candidate's word class and word
take a PP with the dependent's preposition (see pptable.jl); with `pprel`, relative to the best
of the token's other candidates.
"""
function default_specs(; charindex::Union{CharIndex,Nothing} = nothing, wordclasses::Union{WordClasses,Nothing} = nothing,
  suffixes::Vector{Int} = Int[], noise::Int = 0, rootabsent::Bool = false, projective::Bool = false,
  relations::Bool = false, headsdist::Bool = false, pptable = nothing, pprel::Bool = false,
  relbounded::Bool = false, relnone::Bool = false, treerank::Bool = false, headtrank::Bool = false)
  safeats = charindex === nothing ? FeatDef[] : [sa_feats(charindex, :left); sa_feats(charindex, :right)]
  sapaths = charindex === nothing ? FeatPath[] : [FeatPath(:sa_left), FeatPath(:sa_right)]
  chars = LayerSpec{CharLayer}(:char,
    vcat([FeatDef(:char, (st, i) -> UInt32(codepoint(segchar(st, i))) + 1),
        FeatDef(:class, (st, i) -> UInt32(charclass(segchar(st, i))))],
      safeats),
    [RelDef(:left_neighbor, :char, false, (st, i) -> i > 1 ? UInt32(i - 1) : NONE),
      RelDef(:right_neighbor, :char, false, (st, i) -> i < nchars(st) ? UInt32(i + 1) : NONE),
      RelDef(:token_of, :token, false, (st, i) -> isspace(segchar(st, i)) ? NONE : PENDING)],  # set by :tokenize
    [DecisionDef(:gap;
      candidates = gap_candidates,
      features = (st, i, c) -> label_features(st, :char, GAP_TAG, i, c),
      items = (st, i, c) -> label_items(st, :char, GAP_TAG, i),
      allcandidates = gap_allcandidates,
      mask = gap_mask,
      ready = (st, i) -> st.final || i <= nchars(st) - st.edge)],
    [Constraint(:tokenize, (st, i) -> i > 0, tokenize!)],
    vcat([FeatPath(:char), FeatPath(:class),
        FeatPath([:left_neighbor], :char), FeatPath([:left_neighbor], :class),
        FeatPath([:right_neighbor], :char), FeatPath([:right_neighbor], :class),
        FeatPath([:left_neighbor, :left_neighbor], :class), FeatPath([:right_neighbor, :right_neighbor], :class),
        FeatPath([:left_neighbor, :token_of], :form),             # the token just finished, if any
        FeatPath([:left_neighbor, :token_of, :head], :form)],     # ...and its head: parse info reaching chars
      sapaths),
    (st, i) -> i)

  wc = wordclasses
  wlevels = wc === nothing ? Int[] : collect(eachindex(wc.sizes))
  wnames = [wclass_name(wc.sizes[l]) for l in wlevels]
  witemtags = [hash("token." * String(n)) for n in wnames]
  snames = [Symbol(:suffix, n) for n in suffixes]
  nnames = noise > 0 ? [:noise] : Symbol[]
  extra = length(wlevels) + length(suffixes) + length(nnames)
  relf = RelFlags(relbounded, relnone)
  relrels, relpaths, rellink = relations ? relation_rows(relf) : (RelDef[], FeatPath[], Int[])
  linkpaths = [LINK_PATHS; 22 .+ (1:extra); 22 + extra .+ rellink]   # classes, suffixes, noise, then relations: after the 22 below
  tokens = LayerSpec{TokenLayer}(:token,
    vcat([FeatDef(:form, (st, t) -> intern!(st.lex, :form, lowercase(tokstr(st, t)))),
        FeatDef(:caps, (st, t) -> UInt32(capsclass(tokstr(st, t)))),
        FeatDef(:len, (st, t) -> UInt32(min(length(tokstr(st, t)), 10)))],
      [FeatDef(snames[j], (st, t) -> intern!(st.lex, snames[j], last(lowercase(tokstr(st, t)), suffixes[j])))
        for j in eachindex(suffixes)],
      # (position and document length together: position alone recurs across short documents, and
      # absolute position does tell something about attachment)
      [FeatDef(:noise, (st, t) -> UInt32(hash((length(st.doc.text), docpos(st, Int(st.layers[:token][:first_char, t])))) % noise + 1))
        for _ in nnames]),
    vcat([RelDef(:first_char, :char, false, (st, t) -> PENDING),       # given by :tokenize
      RelDef(:last_char, :char, false, (st, t) -> PENDING),
      RelDef(:left_neighbor, :token, true, (st, t) -> adjacent_token(st, t, -1)),
      RelDef(:right_neighbor, :token, true, (st, t) -> adjacent_token(st, t, +1)),
      RelDef(:head, :token, true, (st, t) -> st.layers[:token][:attach, t]),
      RelDef(:sentence, :sentence, false, (st, t) -> PENDING)],   # set by :sentences
      relrels),
    vcat([DecisionDef(:attach;
        candidates = (st, t) -> attach_candidates(st, t; projective),
        features = (st, t, c) -> attach_features(st, t, c, linkpaths),
        items = (st, t, c) -> vcat(attach_items(st, t, c, linkpaths; rootabsent),
          relations ? hypothetical_sibling_items(st, t, c, relf) : UInt64[],
          treerank ? (v = trank_code(tree_rank(st, t, c)); UInt64[item(v, (TRANK_TAG, v))]) : UInt64[],
          pptable === nothing ? UInt64[] :
            pp_items(pptable, st, t, c; cands = pprel ? attach_candidates(st, t; projective) : nothing)),
        allcandidates = attach_allcandidates,
        mask = attach_mask,
        open = true,
        labeled = false),
      DecisionDef(:deprel;
        candidates = (st, t) -> DEPREL_CODES,
        features = (st, t, c) -> label_features(st, :token, DEPREL_TAG, t, c),
        items = (st, t, c) -> label_items(st, :token, DEPREL_TAG, t),
        ready = (st, t) -> st.layers[:token][:attach, t] != PENDING)],
      [DecisionDef(name;
          candidates = (st, t) -> TAG_CODES[name],
          features = (st, t, c) -> label_features(st, :token, TAG_ITEMTAGS[name], t, c),
          items = (st, t, c) -> label_items(st, :token, TAG_ITEMTAGS[name], t))
        for name in keys(TAGSETS)],
      [DecisionDef(name;
          candidates = (st, t) -> head_candidates(st, t, name, projective),
          features = (st, t, c) -> label_features(st, :token, HEAD_ITEMTAGS[name], t, c),
          items = (st, t, c) -> label_items(st, :token, HEAD_ITEMTAGS[name], t),
          mask = head_mask,
          # once the attachment is committed, the backward step derives it (the oracle decides it itself)
          ready = (st, t) -> !st.infer || (!SOFTHEADS[] && st.layers[:token][:attach, t] == PENDING))
        for name in [collect(keys(HEADSETS)); headsdist ? [:head_sdist] : Symbol[]; headtrank ? [:head_trank] : Symbol[]]],
      [DecisionDef(wnames[l];
          candidates = (st, t) -> wclass_candidates(wc, l, lowercase(tokstr(st, t))),
          features = (st, t, c) -> label_features(st, :token, witemtags[l], t, c),
          items = (st, t, c) -> label_items(st, :token, witemtags[l], t))
        for l in wlevels]),
    [Constraint(:sentences, (st, t) -> true, close_sentences!), Constraint(:infer_attach, (st, t) -> t > 0, infer_attach!),
      Constraint(:derive_heads, (st, t) -> t > 0, derive_heads!)],
    vcat([FeatPath(:form), FeatPath(:caps), FeatPath(:len),
        FeatPath([:left_neighbor], :form), FeatPath([:right_neighbor], :form), FeatPath([:head], :form)],
      [FeatPath(name) for name in keys(TAGSETS)],
      [FeatPath([:left_neighbor], :upos), FeatPath([:right_neighbor], :upos),
        FeatPath([:left_neighbor], :xpos), FeatPath([:right_neighbor], :xpos),
        FeatPath([:head], :upos), FeatPath([:head], :xpos)],
      [FeatPath(name) for name in keys(HEADSETS)],           # 19:22, not in LINK_PATHS
      [FeatPath(name) for name in wnames],                   # 23:, in the attachment features
      [FeatPath(name) for name in snames],                   # then the suffixes, also in them
      [FeatPath(name) for name in nnames],                   # then the noise feature
      relpaths,                                              # then the relations
      headsdist ? [FeatPath(:head_sdist)] : FeatPath[],      # then optional head properties (not in LINK_PATHS)
      headtrank ? [FeatPath(:head_trank)] : FeatPath[]),
    (st, t) -> Int(st.layers[:token][:last_char, t]))

  length(tokens.context) == 22 + extra + length(relpaths) + headsdist + headtrank || error("token context paths moved: update linkpaths")
  LINKPATHS[tokens] = linkpaths
  sentences = LayerSpec{SentenceLayer}(:sentence,
    FeatDef[],
    [RelDef(:first_token, :token, false, (st, s) -> PENDING),     # given by :sentences
      RelDef(:last_token, :token, false, (st, s) -> PENDING)],
    DecisionDef[], Constraint[], FeatPath[],
    (st, s) -> Int(st.layers[:token][:last_char, Int(st.layers[:sentence][:last_token, s])]))

  LayerSpec[chars, tokens, sentences]
end
