def edit(p, pairs):
    t = open(p).read()
    for old, new in pairs:
        assert t.count(old) == 1, (p, old[:80], t.count(old)); t = t.replace(old, new)
    open(p, "w").write(t)

edit("src/specs.jl", [
 # kin_between via a tracing version that names the token holding it up
 ('''function kin_between(headof, t::Int, n::Int, c, maxn::Int)
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
end''', '''kin_between(headof, t::Int, n::Int, c, maxn::Int) = first(kin_trace(headof, t, n, c, maxn))

"""
kin_between, also naming the token whose pending attachment holds the answer up: (code, 0), or (PENDING, u) -- u's
head unknown -- or (PENDING, 0) on a cycle (an impossible combination).
"""
function kin_trace(headof, t::Int, n::Int, c, maxn::Int)
  up = Dict{Int,Int}(t => 0)                         # t's ancestors (c first) and their steps from t
  h, d, complete, last, pend = c, 0, true, t, 0
  while true
    h == PENDING && (complete = false; pend = last; break)
    h == NONE && break
    u = Int(h)
    haskey(up, u) && return (PENDING, 0)
    d += 1; up[u] = d
    d > maxn && return (PENDING, 0)
    last = u
    h = headof(u)
  end
  v, j = n, 0
  while true
    haskey(up, v) && return (kincode(up[v], j), 0)
    hv = headof(v)
    hv == PENDING && return (PENDING, v)
    hv == NONE && return complete ? (KIN_OTHER, 0) : (PENDING, pend)
    v = Int(hv); j += 1
    j > maxn && return (PENDING, 0)
  end
end'''),
 # bioes_label with an override for one hidden token
 ('''function bioes_label(st, u::Int, t::Int, c)
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
  end''', '''function bioes_label(st, u::Int, t::Int, c; p::Int = 0, hp = PENDING)
  toks = st.layers[:token]
  ord, rank, cp, sfirst, slast = relranks(st)
  ru, rt = rank[u], rank[t]
  ch = children_index(st)[u]
  # (forward checking: token p's attachment taken to be hp, whatever it is now)
  hasL = any(v -> v != t && v != p && rank[v] < ru, ch) || (c == u && rt < ru) || (p > 0 && hp == u && rank[p] < ru)
  hasR = any(v -> v != t && v != p && rank[v] > ru, ch) || (c == u && rt > ru) || (p > 0 && hp == u && rank[p] > ru)
  function settled(a, b)                             # every token in ranks a..b attached (t, p: hypothetically)
    a > b && return true
    n = cp[b+1] - cp[a]
    a <= rt <= b && toks[:attach, t] == PENDING && (n -= 1)
    p > 0 && p != t && a <= rank[p] <= b && toks[:attach, p] == PENDING && (n -= 1)
    n == 0
  end'''),
 # set items, after bioes_items
 ('''bioes_items(st, t, c) = UInt64[item(v, (BIOES_TAG, j, v)) for (j, v) in enumerate(bioes_values(st, t, c))]''',
  '''bioes_items(st, t, c) = UInt64[item(v, (BIOES_TAG, j, v)) for (j, v) in enumerate(bioes_values(st, t, c))]

##======== forward checking: value sets (default_specs(fwdsets = true); user's design 2026-11-03)
#
# A forward-checked item gets a paired set slot: the values the item can still take, over the possible heads of one
# token whose attachment it hangs on -- the token holding it up while it is pending (decoding), or, once known, the
# neighbour it depends on with its attachment hidden (as in training, where everything is known).  Sets of more than
# FSET_MAX values, or ones another pending attachment would also decide, are left out (absent).

const FSET_MAX = 3
const FSET_TAG = hash("token.attach.fwdset")
const HIDDEN_CACHE = Dict{UInt64,Vector{UInt32}}()

"""
Possible heads of token u with its own attachment hidden: in reach, no cycle, no sentence end between, not crossing
another committed arc (projective; all of them if every one crosses).  Cached per state of the arcs.
"""
function hidden_heads(st, u::Int)
  toks = st.layers[:token]
  n = toks.ncols
  key = hash((objectid(st), st.span, u, UInt32[toks[:attach, v] for v in 1:n], UInt32[toks[:sentence, v] for v in 1:n],
    UInt32[toks[:first_char, v] for v in 1:n]))
  r = get(HIDDEN_CACHE, key, nothing)
  r === nothing || return r
  length(HIDDEN_CACHE) >= 200_000 && empty!(HIDDEN_CACHE)
  s = toks[:sentence, u]
  hasroot = s != PENDING && any(v -> v != u && toks[:sentence, v] == s && toks[:attach, v] == NONE, 1:n)
  cands = hasroot ? UInt32[] : UInt32[NONE]
  fu = Int(toks[:first_char, u])
  for h in 1:n
    h == u && continue
    toks[:sentence, h] == s || continue
    abs(Int(toks[:first_char, h]) - fu) <= MAX_LINK_CHARS || continue
    creates_cycle(st, u, h) && continue
    crosses_E(st, u, h) && continue
    push!(cands, UInt32(h))
  end
  proj = filter(h -> !crosses_arc(st, u, h), cands)
  HIDDEN_CACHE[key] = isempty(proj) ? cands : proj
end

"The values f(h) over token p's hidden heads h, sorted -- nothing if one is PENDING (another attachment decides it) or there are more than FSET_MAX; f returning `skip` drops h (impossible)."
function value_set(f, st, p::Int; skip = nothing)
  vals = UInt32[]
  for h in hidden_heads(st, p)
    v = f(h)
    v === skip && continue
    v == PENDING && return nothing
    v in vals || push!(vals, v)
    length(vals) > FSET_MAX && return nothing
  end
  isempty(vals) ? nothing : Tuple(sort!(vals))
end

"Kinship set of t to its neighbour on side dir under candidate c (see the header above)."
function kin_set(st, t, c, dir)
  c == PENDING && return nothing
  toks = st.layers[:token]
  n = adjacent_token(st, t, dir)
  n == PENDING && return nothing
  n == NONE && return (KIN_NONE,)
  (toks[:sentence, t] == PENDING || toks[:sentence, Int(n)] == PENDING) && return nothing
  toks[:sentence, Int(n)] == toks[:sentence, t] || return (KIN_NONE,)
  nn, N = Int(n), length(toks)
  v, p = kin_trace(u -> toks[:attach, u], t, nn, c, N)
  v == PENDING && p == 0 && return nothing
  p = v == PENDING ? p : nn                          # known: hide the neighbour, as in training
  p == t && return nothing
  value_set(st, p; skip = :cycle) do h
    w, q = kin_trace(u -> u == p ? h : toks[:attach, u], t, nn, c, N)
    w == PENDING ? (q == 0 ? :cycle : PENDING) : w
  end
end

"The unattached tokens (t, and p if given, excluded) on u's unsettled sides -- those a BIOES label of u waits for."
function bioes_waiting(st, u::Int, t::Int)
  toks = st.layers[:token]
  ord, rank, cp, sfirst, slast = relranks(st)
  ru = rank[u]
  [ord[r] for r in sfirst[ru]:slast[ru] if r != ru && ord[r] != t && toks[:attach, ord[r]] == PENDING]
end

"BIOES gap-pair set for t's side dir under candidate c: pairs (left label, right label) as bioes_values codes."
function bioes_gap_set(st, t, c, dir)
  c == PENDING && return nothing
  toks = st.layers[:token]
  n = adjacent_token(st, t, dir)
  n == PENDING && return nothing
  edge = n == NONE || toks[:sentence, Int(n)] != toks[:sentence, t]
  (toks[:sentence, t] == PENDING || (!edge && toks[:sentence, Int(n)] == PENDING)) && return nothing
  pair(a, b) = (a == PENDING || b == PENDING) ? PENDING : UInt32(5 * (a - 1) + b)
  labs(p, hp) = (lt = bioes_label(st, t, t, c; p, hp); ln = edge ? UInt32(5) : bioes_label(st, Int(n), t, c; p, hp);
    dir < 0 ? pair(ln, lt) : pair(lt, ln))
  v = labs(0, PENDING)
  if v == PENDING                                    # pending: the one token both labels wait for, if just one
    w = union(bioes_waiting(st, t, t), edge ? Int[] : bioes_waiting(st, Int(n), t))
    length(w) == 1 || return nothing
    p = only(w)
  else
    edge && return (v,)
    p = Int(n)                                       # known: hide the neighbour, as in training
  end
  value_set(h -> labs(p, h), st, p)
end

"idxeq set j (3, 4: the left / right neighbour attached to c; 5, 6: c's head is the left / right neighbour): 1 same, 2 different."
function idxeq_set(st, t, c, j)
  c == PENDING && return nothing
  nb = adjacent_token(st, t, j in (3, 5) ? -1 : 1)
  nb == PENDING && return nothing
  if j in (3, 4)
    nb == NONE && return (UInt32(2),)
    return value_set(h -> h == c ? UInt32(1) : UInt32(2), st, Int(nb))
  end
  c == NONE && return (UInt32(2),)
  nb == NONE && return (UInt32(2),)
  value_set(h -> h == nb ? UInt32(1) : UInt32(2), st, Int(c))
end

"The set items (see above) for t under candidate c, for the item kinds in use."
function fwd_set_items(st, t, c; kin = false, bioes = false, idxeq = false)
  sets = Any[]
  kin && append!(sets, [kin_set(st, t, c, -1), kin_set(st, t, c, 1)])
  bioes && append!(sets, [bioes_gap_set(st, t, c, -1), bioes_gap_set(st, t, c, 1)])
  idxeq && append!(sets, [idxeq_set(st, t, c, j) for j in 3:6])
  UInt64[s === nothing ? ABSENT : hash((FSET_TAG, j, s)) for (j, s) in enumerate(sets)]
end

"""
Forced attachments (fcommit, set for decoding): infer_attach! commits any pending attachment left with exactly one
possible head once no other head can appear -- not only those narrowed by committed head properties.
"""
const FCOMMIT = Ref(false)'''),
 # infer_attach!: forced commits
 ('''    decided = [name for name in headnames(toks) if toks[name, t] != PENDING]
    isempty(decided) && continue''', '''    decided = [name for name in headnames(toks) if toks[name, t] != PENDING]
    isempty(decided) && !FCOMMIT[] && continue'''),
 # default_specs
 ('''neighbor2::Bool = false, oclass::Bool = false, bioes::Bool = false)''',
  '''neighbor2::Bool = false, oclass::Bool = false, bioes::Bool = false, fwdsets::Bool = false)'''),
 ('''          bioes ? bioes_items(st, t, c) : UInt64[],
''', '''          bioes ? bioes_items(st, t, c) : UInt64[],
          fwdsets ? fwd_set_items(st, t, c; kin = kinitems, bioes, idxeq) : UInt64[],
'''),
])
edit("experiments/dev_eval.jl", [
 ('#   oclass       true:', '''#   fwdsets      true: forward checking -- value-set slots paired with the kinitems, bioes and idxeq items: the values an
#                item can still take over one pending (decoding) or hidden (training) attachment's possible heads,
#                if at most 3; events apart
#   fcommit      true (decode): commit any pending attachment left with one possible head (forward checking)
#   oclass       true:'''),
 ('"oclass" => "false",', '"oclass" => "false", "fwdsets" => "false", "fcommit" => "false",'),
 ('''    oclass = cfg["oclass"] == "true", bioes = cfg["bioes"] == "true")''',
  '''    oclass = cfg["oclass"] == "true", bioes = cfg["bioes"] == "true", fwdsets = cfg["fwdsets"] == "true")'''),
 ('''(cfg["oclass"] == "true" ? "-ocl" : "") * (cfg["bioes"] == "true" ? "-bioes" : ""))''',
  '''(cfg["oclass"] == "true" ? "-ocl" : "") * (cfg["bioes"] == "true" ? "-bioes" : "") * (cfg["fwdsets"] == "true" ? "-fs" : ""))'''),
 ('''(cfg["bioes"] == "true" ? ";bioes" : "") * (cfg["goldtok"] == "true" ? ";goldtok" : "")''',
  '''(cfg["bioes"] == "true" ? ";bioes" : "") * (cfg["fwdsets"] == "true" ? ";fwdsets" : "") *
      (cfg["fcommit"] == "true" ? ";fcommit" : "") * (cfg["goldtok"] == "true" ? ";goldtok" : "")'''),
 ('''      oclass = $(cfg["oclass"] == "true"), bioes = $(cfg["bioes"] == "true"))''',
  '''      oclass = $(cfg["oclass"] == "true"), bioes = $(cfg["bioes"] == "true"), fwdsets = $(cfg["fwdsets"] == "true"))'''),
 ('''    $(cfg["fwdcheck"] == "true") && (SeqLayers.FWDCHECK[] = true)''', '''    $(cfg["fwdcheck"] == "true") && (SeqLayers.FWDCHECK[] = true)
    $(cfg["fcommit"] == "true") && (SeqLayers.FCOMMIT[] = true)'''),
])
print("ok")
