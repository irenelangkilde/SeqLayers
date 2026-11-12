# ceiling_lib.jl  --  perfect-context scoring, loaded by ceiling.jl and its workers
#
# A dev doc is run through the gold oracle.  At the end of each segment, every committed gap,
# attach and deprel in a closed sentence is masked as in events=gold recording (only it and
# what's derived from it hidden; everything else at its gold value), re-decided by the model
# from its usual (constrained) candidates, and compared to gold.  With head-property decisions in
# the specs, an attachment is scored as a pipeline instead: hide the attachment and all of the
# token's head-property decisions, predict those, commit the ones the model is at least `headconf`
# sure of (as easy-first decoding would: unsure ones stay pending), let them narrow the candidates
# (strictly), infer the attachment if one candidate is left, else let the model choose.  With
# `headgold`, the named head properties are given at their gold values instead of predicted: a
# ceiling for the attachment mechanism (narrowing, inference, then the attachment choice).  With
# `domains` (:prune or :product), no head property is committed: the attachment and head
# properties keep domains (values with probability >= `domeps`) narrowed by propagation, and the
# attachment is chosen over its domain (src/domains.jl).

using SeqLayers
import SeqLayers: observe!, segment_done!, segment_discard!

"(gap_trace.jl) called with (state, token, gold head, predicted head, gold deprel, predicted deprel) per scored attachment."
const TOKEN_HOOK = Ref{Any}(nothing)

const CEILING_COUNTS = (:tok_gold, :gap_n, :gap_ok, :gap_lost, :tok_n, :uas, :las, :head_lost, :notready,
  :inferred, :inferred_ok, :nocands)

mutable struct CeilingRecorder
  model::Any
  doc::Doc
  scoretags::Bool                                # also score the tag decisions (slow)
  headconf::Float64                              # pipeline: commit a predicted head property only this sure
  headgold::Vector{Symbol}                       # pipeline: head properties given at their gold values
  c::Dict{Symbol,Int}
  parts::Union{Nothing,Vector{Any}}              # calibration: per decision (type, chain log odds, corrections, wait, gold index)
  domains::Symbol                                # pipeline with domain variables: :off | :prune | :product
  domeps::Float64
end
CeilingRecorder(model, doc, scoretags, headconf, headgold = Symbol[]; collect = false, domains = :off, domeps = 1e-3) =
  CeilingRecorder(model, doc, scoretags, headconf, headgold, Dict(k => 0 for k in CEILING_COUNTS), collect ? Any[] : nothing, domains, domeps)

"""
`decide`, recording for calibration (when r.parts is set) each candidate's chain log odds and leftover
correction, the wait score of open decisions, and where gold is among the candidates.
"""
function rdecide(r::CeilingRecorder, st, key, gold)
  r.parts === nothing && return decide(r.model, st, key)
  empty!(r.model.memo)                           # every decision scored afresh, so the hook sees it
  buf = Tuple{Float64,Float64}[]
  SeqLayers.PARTS_HOOK[] = (lo, c) -> push!(buf, (lo, c))
  res = try
    decide(r.model, st, key)
  finally
    SeqLayers.PARTS_HOOK[] = nothing
  end
  d = SeqLayers.decisiondef(st, key)
  cands = d.ready(st, key[3]) ? d.candidates(st, key[3]) : UInt32[]
  g = findfirst(==(gold), cands)
  if g !== nothing && length(cands) >= 2 && length(buf) == length(cands)
    w = d.open ? SeqLayers.waitscore(r.model, st, key) : NaN
    push!(r.parts, (SeqLayers.typekey(key[1], key[2]), first.(buf), last.(buf), w, g))
  end
  res
end

observe!(::CeilingRecorder, st, key, choice) = nothing
segment_discard!(::CeilingRecorder) = nothing

"The model's choice for `key` with only that decision masked, and whether gold was among the candidates (NONE is a valid choice: root)."
function masked_choice(r::CeilingRecorder, st, key, gold::UInt32)
  d, i = SeqLayers.decisiondef(st, key), key[3]
  restore = d.mask(st, key)
  try
    d.ready(st, i) || (r.c[:notready] += 1; return (PENDING, false))
    cands = d.candidates(st, i)
    isempty(cands) && return (PENDING, false)
    first(rdecide(r, st, key, gold)), gold in cands
  finally
    restore()
  end
end

function segment_done!(r::CeilingRecorder, st)
  chars, toks = st.layers[:char], st.layers[:token]
  for i in 1:chars.ncols
    g = chars[:gap, i]
    (g == PENDING || g == NONE || i > st.closed_through) && continue
    p, incands = masked_choice(r, st, (:char, 1, i), g)
    r.c[:gap_n] += 1
    r.c[:gap_ok] += p == g
    r.c[:gap_lost] += !incands
  end
  for t in 1:toks.ncols
    a, rel = toks[:attach, t], toks[:deprel, t]
    (a == PENDING || rel == PENDING || Int(toks[:last_char, t]) > st.closed_through) && continue
    haskey(r.doc.gold.head, SeqLayers.tokspan(st, t)) || continue     # misaligned with gold
    pa, incands = any(dd -> haskey(SeqLayers.ALLHEADSETS, dd.name), toks.spec.decisions) ?
      pipeline_attach!(r, st, t, a) : masked_choice(r, st, (:token, 1, t), a)
    pr, _ = masked_choice(r, st, (:token, 2, t), rel)
    r.c[:tok_n] += 1
    r.c[:head_lost] += !incands
    r.c[:uas] += pa == a
    r.c[:las] += pa == a && pr == rel
    TOKEN_HOOK[] === nothing || TOKEN_HOOK[](st, t, a, pa, rel, pr)
    breakdown!(r.c, st, t, a, rel, pa, pr)
    r.scoretags || continue
    for k in 3:length(toks.spec.decisions)        # tag decisions (head properties: scored in the pipeline)
      haskey(SeqLayers.ALLHEADSETS, toks.spec.decisions[k].name) && continue
      name = toks.spec.decisions[k].name
      g = toks[name, t]
      g == PENDING && continue
      p, _ = masked_choice(r, st, (:token, k, t), g)
      for (m, x) in (("n", true), ("ok", p == g))
        key = Symbol("tag|", name, "|", m)
        r.c[key] = get(r.c, key, 0) + x
      end
    end
  end
end

"""
The attachment of token t as a pipeline (see the header): its head-property decisions predicted
with all of them and the attachment hidden, then the attachment from the narrowed candidates.
Returns (choice, whether gold was among the candidates); also counts head-property accuracy.
"""
function pipeline_attach!(r::CeilingRecorder, st, t, gold::UInt32)
  toks = st.layers[:token]
  decs = toks.spec.decisions
  hk = [k for (k, d) in enumerate(decs) if haskey(SeqLayers.ALLHEADSETS, d.name)]
  names = [:attach; [decs[k].name for k in hk]]
  old = [toks[n, t] for n in names]
  foreach(n -> (toks[n, t] = PENDING), names)
  SeqLayers.refresh!(st)
  r.domains == :off || return domains_attach!(r, st, t, gold, hk, old, names)
  try
    res = [decs[k].name in r.headgold ? (old[j+1], Inf) : rdecide(r, st, (:token, k, t), old[j+1]) for (j, k) in enumerate(hk)]
    # the separate direction and distance predictions both right (compare head_sdist's accuracy)
    jd, jt = findfirst(k -> decs[k].name == :head_dir, hk), findfirst(k -> decs[k].name == :head_dist, hk)
    if jd !== nothing && jt !== nothing
      ok = res[jd][1] == old[jd+1] && res[jt][1] == old[jt+1]
      for (m, x) in (("n", true), ("ok", ok))
        key = Symbol("tag|head_dir+head_dist|", m)
        r.c[key] = get(r.c, key, 0) + x
      end
    end
    minmargin = log(r.headconf) - log1p(-r.headconf)          # probability -> log-odds
    for (j, k) in enumerate(hk)
      p, margin = res[j]
      for (m, x) in (("n", true), ("ok", p == old[j+1]))
        key = Symbol("tag|", decs[k].name, "|", m)
        r.c[key] = get(r.c, key, 0) + x
      end
      if margin >= minmargin
        toks[decs[k].name, t] = p
        for (m, x) in (("n", true), ("ok", p == old[j+1]))
          key = Symbol("tag|", decs[k].name, "@committed|", m)
          r.c[key] = get(r.c, key, 0) + x
        end
      end
    end
    SeqLayers.refresh!(st)
    cands = SeqLayers.decisiondef(st, (:token, 1, t)).candidates(st, t)
    if length(cands) == 1
      r.c[:inferred] += 1
      r.c[:inferred_ok] += only(cands) == gold
      return only(cands), gold in cands
    end
    isempty(cands) && (r.c[:nocands] += 1; return PENDING, false)
    first(rdecide(r, st, (:token, 1, t), gold)), gold in cands
  finally
    foreach(((n, v),) -> (toks[n, t] = v), zip(names, old))
    SeqLayers.refresh!(st)
  end
end

"""
pipeline_attach! with domain variables (r.domains): every head property and the attachment
predicted with all of them hidden, as distributions; their domains narrowed together; the
attachment chosen over its domain (SeqLayers.domain_choice).  Head properties in r.headgold are
given: a one-value domain.
"""
function domains_attach!(r::CeilingRecorder, st, t, gold::UInt32, hk, old, names)
  toks = st.layers[:token]
  decs = toks.spec.decisions
  try
    props = Dict{Symbol,Tuple{Vector{UInt32},Vector{Float64}}}()
    for (j, k) in enumerate(hk)
      name = decs[k].name
      if name in r.headgold
        props[name] = ([old[j+1]], [1.0])
        continue
      end
      dist = SeqLayers.commit_dist(r.model, st, (:token, k, t))
      dist === nothing && continue
      vals, lps = dist
      ps = exp.(lps)
      props[name] = (vals, ps)
      for (m, x) in (("n", true), ("ok", vals[argmax(ps)] == old[j+1]))
        key = Symbol("tag|", name, "|", m)
        r.c[key] = get(r.c, key, 0) + x
      end
    end
    d = SeqLayers.decisiondef(st, (:token, 1, t))
    d.ready(st, t) || (r.c[:notready] += 1; return PENDING, false)
    cands = d.candidates(st, t)
    isempty(cands) && (r.c[:nocands] += 1; return PENDING, false)
    length(cands) == 1 && return only(cands), gold in cands
    dist = SeqLayers.commit_dist(r.model, st, (:token, 1, t))
    dist === nothing && return first(rdecide(r, st, (:token, 1, t), gold)), gold in cands
    acands, alps = dist
    alive, D = SeqLayers.propagate_domains(st, t, acands, exp.(alps), props; eps = r.domeps)
    priors = SeqLayers.PROD_NORM[] == :prior ?      # (prodnorm=prior) each head property's training prior over its values
      Dict(decs[k].name => SeqLayers.value_priors(r.model, k, props[decs[k].name][1]) for k in hk if haskey(props, decs[k].name)) : nothing
    comps = PRODDUMP[] == "" ? nothing : Any[]
    choice, nalive = SeqLayers.domain_choice(st, t, acands, exp.(alps), props; eps = r.domeps, mode = r.domains, priors, comps)
    comps === nothing || dump_prod(comps, gold)
    nalive == 1 && (r.c[:inferred] += 1; r.c[:inferred_ok] += choice == gold)
    r.c[:dom_cands] = get(r.c, :dom_cands, 0) + length(acands)
    r.c[:dom_alive] = get(r.c, :dom_alive, 0) + nalive
    choice, gold in acands[alive]
  finally
    foreach(((n, v),) -> (toks[n, t] = v), zip(names, old))
    SeqLayers.refresh!(st)
  end
end

"(proddump) This worker's file prefix for the product's per-candidate terms; empty: off.  Set by ceiling.jl."
const PRODDUMP = Ref("")
const PRODDUMP_N = Ref(0)
const PRODDUMP_NAMES = [:attach; sort(collect(keys(SeqLayers.ALLHEADSETS)))]

"One attachment's surviving candidates: decision id, is-gold, then each term (unweighted; NaN: not in the product)."
function dump_prod(comps, gold)
  PRODDUMP_N[] += 1
  open(PRODDUMP[] * "-w$(getpid()).tsv", "a") do io
    for (c, cj) in comps
      println(io, join([string(PRODDUMP_N[]), c == gold ? "1" : "0", [string(get(cj, n, NaN)) for n in PRODDUMP_NAMES]...], '\t'))
    end
  end
end

"Signed word distance from token t to head h within the sentence (0 = root); negative = head on the left."
function word_dist(st, t, h)
  h == NONE && return 0
  toks = st.layers[:token]
  ft, fh = Int(toks[:first_char, t]), Int(toks[:first_char, h])
  lo, hi = minmax(ft, fh)
  s = toks[:sentence, t]
  between = count(u -> toks[:sentence, u] == s && lo < Int(toks[:first_char, u]) < hi, 1:toks.ncols)
  sign(fh - ft) * (between + 1)
end

"Training-sample form frequencies (lowercased forms), set on each worker by ceiling.jl; empty: no freq dims."
const FORMFREQ = Ref(Dict{String,Int}())
freqclass(n) = n == 0 ? "0" : n < 5 ? "1-4" : n < 50 ? "5-49" : "50+"
formfreq(st, t) = get(FORMFREQ[], lowercase(SeqLayers.tokstr(st, t)), 0)

distclass(d) = d == 0 ? "root" : (d < 0 ? "L" : "R") * (abs(d) == 1 ? "1" : abs(d) == 2 ? "2" : abs(d) <= 5 ? "3-5" : "6+")

"Per-category counts, keyed dim|value|{n,uas,las}; errors also by kind (dim err)."
function breakdown!(c, st, t, a, rel, pa, pr)
  ok, lok = pa == a, pa == a && pr == rel
  gd = word_dist(st, t, a)
  dims = [("dist", distclass(gd)), ("deprel", SeqLayers.DEPRELS[rel])]
  if !isempty(FORMFREQ[])            # how often the dependent's and the gold head's forms occur in training
    push!(dims, ("depfreq", freqclass(formfreq(st, t))))
    push!(dims, ("headfreq", a == NONE ? "root" : freqclass(formfreq(st, Int(a)))))
  end
  for (dim, v) in dims
    for (m, x) in (("n", true), ("uas", ok), ("las", lok))
      k = Symbol(dim, "|", v, "|", m)
      c[k] = get(c, k, 0) + x
    end
  end
  ok && return
  pd = pa == PENDING ? nothing : word_dist(st, t, pa)
  kind = pd === nothing ? "no_choice" : a == NONE ? "missed_root" : pa == NONE ? "false_root" :
    sign(pd) != sign(gd) ? "wrong_side" : abs(pd) < abs(gd) ? "too_near" : abs(pd) > abs(gd) ? "too_far" : "other"
  k = Symbol("err|", kind, "|n")
  c[k] = get(c, k, 0) + 1
end

"Perfect-context counts for one dev doc."
function ceiling_counts(doc::Doc, specs, model, lex; scoretags = true, headconf = 0.0, headgold = Symbol[], collect = false,
  domains = :off, domeps = 1e-3, kw...)
  r = CeilingRecorder(model, doc, scoretags, headconf, headgold; collect, domains, domeps)
  dc = SeqLayers.DC_STATS[]
  dc === nothing || empty!(dc)
  process(doc, specs, OracleModel(), lex; observer = r, kw...)
  r.c[:tok_gold] = length(doc.gold.head)
  dc === nothing || for (k, v) in dc                 # double-counting counter (dcstats=true), in thousandths
    r.c[k] = round(Int, 1000v)
  end
  collect ? (r.c, r.parts) : r.c
end
