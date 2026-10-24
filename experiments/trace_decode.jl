# trace_decode.jl  --  decode one text with a chain model and print every committed decision in the
# order the easy-first agenda made it: what, where, the value chosen, and how confident it was.
#
#   julia --project=experiments experiments/trace_decode.jl text="I ran across the item in the internet" [dev_eval.jl options]
#
#   explain=item:number,xpos   for these decisions of that token, also print each candidate value's
#                              chain (feature = value, odds after each step), leftovers (tried, included,
#                              context, correction) and final score, scored just before the commit
#   explaintop=60              show at most this many candidate values, best first
#   explainat=head_upos        also explain the listed decisions, while still open, when this decision
#                              of the token commits (e.g. an attachment later inferred, never decided)
#
# Confidence: the chosen value's probability among the decision's candidates (choose's softmax, with
# "wait" for open decisions) and its margin in log-odds; Inf: the only candidate.  Decisions settled by
# constraints (tokenization, sentences, inferred attachments, derived head properties) are listed as
# such when they appear.

include(joinpath(@__DIR__, "dev_eval.jl"))
const S = SeqLayers
import SeqLayers: observe!, segment_done!, segment_discard!

DEFAULTS["text"] = "I ran across the item in the internet."
DEFAULTS["explain"] = ""
DEFAULTS["explaintop"] = 60
DEFAULTS["explainat"] = ""
cfg = parse_cfg(["model=chain"; ARGS])
s = setup(cfg)
idx = s.idx
PPT = isempty(cfg["pptable"]) ? nothing : load_pptable(cfg["pptable"], joinpath(dirname(dirname(cfg["pptable"])), "classes5"))
widx = CharIndex(idx.fwd, idx.rev, idx.gap, idx.saL, idx.saR, idx.cumL, idx.cumR, idx.prior, IdDict{Doc,Int}(), idx.maxlen, idx.mincount, idx.beta)
specs = default_specs(charindex = widx, wordclasses = s.wc, suffixes = s.sfx, noise = cfg["noise"], rootabsent = cfg["rootabsent"] == "true",
  projective = cfg["projective"] == "true", relations = cfg["relations"] == "true", headsdist = cfg["headsdist"] == "true",
  pptable = PPT)
eventsdir, opts = s.recipe
m = ChainModel(eventsdir; opts...)
pc = S.load_counts(eventsdir, eventsig(eventsdir))
pc !== nothing && (m.pcounts = pc)
get(cfg, "poolsigma", 0.0) > 0 && (m.generalize = S.generalization_slots(specs))

mutable struct TraceRec
  n::Int
  rows::Vector{String}
end
tokname(st, t) = t == NONE ? "ROOT" : t == PENDING ? "?" : S.tokstr(st, Int(t))
function valname(st, lname, name, v)
  lname == :char && return S.GAP_NAMES[v]
  name == :attach && return "-> " * tokname(st, v)
  name == :deprel && return S.DEPRELS[v]
  haskey(S.TAGSETS, name) && return S.TAGSETS[name][v]
  haskey(S.ALLHEADSETS, name) && return S.ALLHEADSETS[name][v]
  string(v)
end
ctxname(p) = join([string.(p.rels); string(p.feat)], '.')
"Name of the value of token context path p for token t."
function ctxval(st, t, p)
  v = S.follow(st, :token, t, p)
  v == PENDING && return "?"
  v == NONE && return "none"
  f = p.feat
  f == :form && return get(get(st.lex.names, :form, String[]), Int(v), string(v))
  haskey(S.TAGSETS, f) && return S.TAGSETS[f][v]
  haskey(S.ALLHEADSETS, f) && return S.ALLHEADSETS[f][v]
  f == :deprel && return S.DEPRELS[v]
  string(v)
end
"Name of slot s of decision d's records for token t and candidate c (attachment: dependent/candidate sides)."
function slotdesc(st, d, t, c, s)
  ctx = st.layers[:token].spec.context
  if d.name == :attach
    s == 1 && return "distance"
    s == 2 && return "direction"
    lp = S.LINKPATHS[st.layers[:token].spec]
    a = (s - 3) ÷ 2 + 1
    n = length(d.items(st, t, c))
    PPT !== nothing && s > n - 2 && return ppdesc(st, t, c, s - (n - 2))
    a > length(lp) && return "slot $s"
    side, u = isodd(s - 3) ? ("cand", c) : ("dep", UInt32(t))
    return "$side.$(ctxname(ctx[lp[a]]))=$(u == NONE ? "none" : ctxval(st, Int(u), ctx[lp[a]]))"
  end
  s <= length(ctx) || return "slot $s"
  ctxname(ctx[s]) * "=" * ctxval(st, t, ctx[s])
end
"Name and bucket of the PP table item k (1: candidate's class rate, 2: its word's rate) for t -> c."
function ppdesc(st, t, c, k)
  name = k == 1 ? "cand.ppclass" : "cand.ppword"
  p = c == S.NONE ? nothing : S.case_prep(st, t)
  (p === nothing || !haskey(PPT.base, p)) && return "$name=absent"
  w = lowercase(S.tokstr(st, Int(c)))
  if k == 1
    cl = get(PPT.formclass, w, Int32(0))
    (cl == 0 || !haskey(PPT.classocc, cl)) && return "$name($p)=unknown"
    npp, nocc = get(PPT.classpp, (cl, p), 0.0), PPT.classocc[cl]
  else
    haskey(PPT.formocc, w) || return "$name($p)=unknown"
    npp, nocc = get(PPT.formpp, (w, p), 0.0), PPT.formocc[w]
  end
  b = S.pp_bucket(npp, nocc, PPT.base[p])
  rate = round(100 * (npp + 20 * PPT.base[p]) / (nocc + 20); sigdigits = 2)
  "$name($p)=$b ($rate per 100)"
end

function explain(st, key)
  lname, k, t = key
  d = st.layers[lname].spec.decisions[k]
  cands = d.candidates(st, t)
  items = [d.items(st, t, c) for c in cands]
  slotof(c, x) = findfirst(==(x), items[findfirst(==(c), cands)])
  events = Dict{Any,Any}(c => (left = Any[], final = nothing) for c in cands)
  S.CHAIN_TRACE[] = (ev, data) -> ev == :final && (events[data.cand] = (left = events[data.cand].left, final = data))
  S.LEFTOVER_TRACE[] = (slot, inc, ctxs, corr) -> push!(events[S.CHAIN_TRACE_CAND[]].left, (slot, inc, ctxs, corr))
  empty!(m.memo)
  try
    decide(m, st, key)
  finally
    S.CHAIN_TRACE[] = nothing; S.LEFTOVER_TRACE[] = nothing
  end
  scores = [events[c].final === nothing ? NaN : events[c].final.logodds for c in cands]
  mx = maximum(filter(!isnan, scores); init = 0.0)
  z = sum(x -> isnan(x) ? 0.0 : exp(x - mx), scores)
  println("\n==== $(d.name) of '$(S.tokstr(st, t))': $(length(cands)) candidate values" *
    (length(cands) > cfg["explaintop"] ? " (the $(cfg["explaintop"]) best shown)" : ""))
  for ci in sortperm(scores; rev = true, by = x -> isnan(x) ? -Inf : x)[1:min(end, cfg["explaintop"])]
    c = cands[ci]
    f = events[c].final
    f === nothing && (println("  $(valname(st, lname, d.name, c)): (no chain)"); continue)
    lab = d.labeled ? c : UInt32(0)
    println("\n  -- $(valname(st, lname, d.name, c)):  final log odds $(round(f.logodds; digits = 2)), probability $(round(exp(f.logodds - mx) / z; digits = 4))")
    println("     prior odds $(round(f.odds[1]; sigdigits = 3))")
    for (j, x) in enumerate(f.members)
      s = slotof(c, x)
      println("     chain $j: ", rpad(s === nothing ? "?" : slotdesc(st, d, t, c, s), 34), " odds -> $(round(f.odds[j+1]; sigdigits = 3))")
    end
    chainlo = log(f.odds[end])
    inc = [l for l in events[c].left if l[2]]
    println("     chain log odds $(round(chainlo; digits = 2)); leftovers tried $(length(events[c].left)), included $(length(inc)), total correction $(round(sum(l -> l[4], inc; init = 0.0); digits = 2))")
    for (slot, _, ctxs, corr) in inc
      println("       + ", rpad(slotdesc(st, d, t, c, slot), 30), lpad(round(corr; digits = 3), 8), "   context: ", join([slotdesc(st, d, t, c, q) for q in ctxs if q > 0], ", "))
    end
  end
end

function observe!(r::TraceRec, st, key, choice)
  lname, k, i = key
  d = st.layers[lname].spec.decisions[k]
  ex = cfg["explain"]
  if !isempty(ex) && lname == :token
    w, names = split(ex, ':')
    S.tokstr(st, i) == w && string(d.name) in split(names, ',') && explain(st, key)
    at = cfg["explainat"]                            # also explain those decisions (still open) when this one commits
    if !isempty(at) && S.tokstr(st, i) == w && string(d.name) == at
      ds = st.layers[lname].spec.decisions
      for nm in split(names, ',')
        k2 = findfirst(x -> string(x.name) == nm, ds)
        k2 === nothing || k2 == k || explain(st, (lname, k2, i))
      end
    end
  end
  _, margin = decide(m, st, key)                     # (the agenda's own score for this commit)
  p = margin == Inf ? 1.0 : 1 / (1 + exp(-margin))
  where_ = lname == :char ? "after '" * string(S.segchar(st, i)) * "' (char $i)" : "'" * S.tokstr(st, i) * "'"
  r.n += 1
  push!(r.rows, rpad(r.n, 5) * rpad(string(d.name), 12) * rpad(where_, 22) * rpad(valname(st, lname, d.name, choice), 14) *
    lpad(string(round(p; digits = 4)), 8) * lpad(margin == Inf ? "Inf" : string(round(margin; digits = 2)), 9))
end
segment_done!(::TraceRec, st) = nothing
segment_discard!(::TraceRec) = nothing

doc = doc_from_text(cfg["text"])
r = TraceRec(0, String[])
out = process(doc, specs, m, s.lex; observer = r)
println("\n\"$(cfg["text"])\"  --  $(s.modelname)\n")
println(rpad("#", 5), rpad("decision", 12), rpad("where", 22), rpad("chosen", 14), lpad("prob", 8), lpad("margin", 9))
foreach(println, r.rows)
println("\nresult:")
for sent in out
  for (j, w) in enumerate(sent.tokens)
    h = sent.heads[j]
    println("  ", rpad(j, 3), rpad(w, 12), "head ", rpad(h == 0 ? "ROOT" : "$(h) $(sent.tokens[h])", 14), sent.deprels[j])
  end
end
