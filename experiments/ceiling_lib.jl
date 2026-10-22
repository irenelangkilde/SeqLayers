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
# ceiling for the attachment mechanism (narrowing, inference, then the attachment choice).

using SeqLayers
import SeqLayers: observe!, segment_done!, segment_discard!

const CEILING_COUNTS = (:tok_gold, :gap_n, :gap_ok, :gap_lost, :tok_n, :uas, :las, :head_lost, :notready,
  :inferred, :inferred_ok, :nocands)

mutable struct CeilingRecorder
  model::Any
  doc::Doc
  scoretags::Bool                                # also score the tag decisions (slow)
  headconf::Float64                              # pipeline: commit a predicted head property only this sure
  headgold::Vector{Symbol}                       # pipeline: head properties given at their gold values
  c::Dict{Symbol,Int}
end
CeilingRecorder(model, doc, scoretags, headconf, headgold = Symbol[]) =
  CeilingRecorder(model, doc, scoretags, headconf, headgold, Dict(k => 0 for k in CEILING_COUNTS))

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
    first(decide(r.model, st, key)), gold in cands
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
  try
    res = [decs[k].name in r.headgold ? (old[j+1], Inf) : decide(r.model, st, (:token, k, t)) for (j, k) in enumerate(hk)]
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
    first(decide(r.model, st, (:token, 1, t))), gold in cands
  finally
    foreach(((n, v),) -> (toks[n, t] = v), zip(names, old))
    SeqLayers.refresh!(st)
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
function ceiling_counts(doc::Doc, specs, model, lex; scoretags = true, headconf = 0.0, headgold = Symbol[], kw...)
  r = CeilingRecorder(model, doc, scoretags, headconf, headgold)
  process(doc, specs, OracleModel(), lex; observer = r, kw...)
  r.c[:tok_gold] = length(doc.gold.head)
  r.c
end
