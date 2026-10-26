# ceiling_explain.jl  --  explain attachment errors of the perfect-context ceiling (as ceiling.jl:
# everything else at its gold value; the token's attachment and head properties hidden and predicted).
#
#   julia --project=experiments experiments/ceiling_explain.jl [dev_eval.jl / ceiling.jl options] \
#     docs=3 nsent=2 maxerr=2 maxlen=16 explaintop=2
#
# Goes through the `docs` smallest slice docs; for the first `nsent` sentences of at most `maxlen` words
# with 1..`maxerr` wrong heads that the head-property ceiling (gold head properties) gets right, prints the sentence (gold vs predicted heads) and, for each wrong head:
# the head-property predictions (top values), then for the gold head and the predicted head (and the
# best others, up to `explaintop` in all): the chain (feature = value, odds after each step), the
# included leftovers with their contexts and corrections, the chain + leftover log odds, the
# attachment probability, and with domains=product the head-property terms of the product score.
# chainonly=true: only errors the chains alone already make (scoring every candidate by its chain log odds,
# without leftovers -- and with domains=product, times the head-property terms -- gives the same wrong head).

include(joinpath(@__DIR__, "dev_eval.jl"))
include(joinpath(@__DIR__, "ceiling_lib.jl"))
const S = SeqLayers
import SeqLayers: observe!, segment_done!, segment_discard!

DEFAULTS["docs"] = 3
DEFAULTS["nsent"] = 2
DEFAULTS["maxerr"] = 2
DEFAULTS["maxlen"] = 16
DEFAULTS["explaintop"] = 2
DEFAULTS["chainonly"] = "false"
DEFAULTS["headconf"] = 1.0
DEFAULTS["domains"] = "off"
DEFAULTS["domeps"] = 1e-3
cfg = parse_cfg(["model=chain"; ARGS])
s = setup(cfg)
idx = s.idx
widx = CharIndex(idx.fwd, idx.rev, idx.gap, idx.saL, idx.saR, idx.cumL, idx.cumR, idx.prior, IdDict{Doc,Int}(), idx.maxlen, idx.mincount, idx.beta)
specs = default_specs(charindex = widx, wordclasses = s.wc, suffixes = s.sfx, noise = cfg["noise"], rootabsent = cfg["rootabsent"] == "true",
  projective = cfg["projective"] == "true", relations = cfg["relations"] == "true", headsdist = cfg["headsdist"] == "true", relbounded = cfg["relbounded"] == "true", relnone = cfg["relnone"] == "true", treerank = cfg["treerank"] == "true", headtrank = cfg["headtrank"] == "true")
eventsdir, opts = s.recipe
m = ChainModel(eventsdir; opts...)
pc = S.load_counts(eventsdir, eventsig(eventsdir))
pc !== nothing && (m.pcounts = pc)
(cfg["poolsigma"] > 0 || cfg["ctxmode"] != "greedy") && (m.generalize = S.generalization_slots(specs))
cfg["classchain"] == "true" || (m.nochain = S.wclass_slots(specs))
cfg["ctxscope"] == "heads" && (m.ctxtypes = Set(S.typekey(:token, k) for (k, d) in enumerate(specs[2].decisions) if k == 1 || haskey(S.ALLHEADSETS, d.name)))
domains, domeps = Symbol(cfg["domains"]), cfg["domeps"]

# names (as trace_decode.jl)
tokname(st, t) = t == NONE ? "ROOT" : S.tokstr(st, Int(t))
ctxname(p) = join([string.(p.rels); string(p.feat)], '.')
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
function slotdesc(st, t, c, sl)
  ctx = st.layers[:token].spec.context
  if sl <= 2                                       # distance (word bucket) and direction, with their values for this candidate
    c == NONE && return sl == 1 ? "distance=root" : "direction=root"
    d = word_dist(st, t, Int(c))
    sl == 2 && return "direction=" * (d < 0 ? "left" : "right")
    a = abs(d)
    return "distance=$(a) word" * (a == 1 ? "" : "s") * " (bucket " * (a == 1 ? "1" : a == 2 ? "2" : a <= 5 ? "3-5" : a <= 10 ? "6-10" : "11+") * ")"
  end
  lp = S.LINKPATHS[st.layers[:token].spec]
  a = (sl - 3) ÷ 2 + 1
  if a > length(lp)
    k = sl - 2 - 2 * length(lp) - (cfg["relations"] == "true" ? 14 : 0)
    cfg["treerank"] == "true" && k == 1 && return "cand.treerank=" * (c == NONE ? "root" : string(S.tree_rank(st, t, c)))
    return "slot $sl"
  end
  side, u = isodd(sl - 3) ? ("cand", c) : ("dep", UInt32(t))
  "$side.$(ctxname(ctx[lp[a]]))=$(u == NONE ? "none" : ctxval(st, Int(u), ctx[lp[a]]))"
end
headval(name, v) = v == PENDING ? "?" : S.ALLHEADSETS[name][v]

"Explain the attachment of t (its attachment and head properties already hidden) with gold head `gold`."
function explain_attach(st, t, gold, pred)
  toks = st.layers[:token]
  decs = toks.spec.decisions
  key = (:token, 1, t)
  hk = [k for (k, d) in enumerate(decs) if haskey(S.ALLHEADSETS, d.name)]
  props = Dict{Symbol,Tuple{Vector{UInt32},Vector{Float64}}}()
  println("    head properties predicted (with the attachment and all of them hidden):")
  for k in hk
    name = decs[k].name
    dist = S.commit_dist(m, st, (:token, k, t))
    dist === nothing && continue
    vals, lps = dist
    props[name] = (vals, exp.(lps))
    order = sortperm(lps; rev = true)[1:min(3, end)]
    println("      ", rpad(name, 10), join(["$(headval(name, vals[j])) $(round(exp(lps[j]); digits = 3))" for j in order], ", "),
      "   (gold head: $(headval(name, S.head_value(st, t, gold, name))), predicted head: $(headval(name, S.head_value(st, t, pred, name))))")
  end
  cands = decs[1].candidates(st, t)
  items = [decs[1].items(st, t, c) for c in cands]
  slotof(c, x) = findfirst(==(x), items[findfirst(==(c), cands)])
  events = Dict{Any,Any}(c => (left = Any[], final = nothing) for c in cands)
  S.CHAIN_TRACE[] = (ev, data) -> ev == :final && (events[data.cand] = (left = events[data.cand].left, final = data))
  S.LEFTOVER_TRACE[] = (sl, inc, ctxs, corr) -> push!(events[S.CHAIN_TRACE_CAND[]].left, (sl, inc, ctxs, corr))
  empty!(m.memo)
  try
    decide(m, st, key)
  finally
    S.CHAIN_TRACE[] = nothing; S.LEFTOVER_TRACE[] = nothing
  end
  dist = S.commit_dist(m, st, key)
  pa = dist === nothing ? Dict{UInt32,Float64}() : Dict(zip(dist[1], exp.(dist[2])))
  # the product score (domains=product, no pruning: domeps applies as in the run)
  prod_ = Dict{UInt32,Float64}()
  if domains == :product && dist !== nothing
    for c in cands
      sc = log(max(get(pa, c, 0.0), 1e-12))
      for (name, (vals, ps)) in props
        v = S.head_value(st, t, c, name)
        v == PENDING && continue
        k = findfirst(==(v), vals)
        sc += log(max(k === nothing ? 0.0 : ps[k], 1e-12))
      end
      prod_[c] = sc
    end
  end
  score(c) = isempty(prod_) ? get(pa, c, 0.0) : get(prod_, c, -Inf)
  others = [c for c in sort(cands; by = c -> -score(c)) if c != gold && c != pred]
  show = unique([gold; pred; others])[1:min(end, max(2, cfg["explaintop"]))]
  println("    candidates: $(length(cands))", isempty(prod_) ? "" : "   (product score = log P(attach) + Σ log P(head property value))")
  for c in show
    f = events[c].final
    tag = c == gold ? "GOLD" : c == pred ? "PREDICTED" : "other"
    println("\n    -- $tag -> $(tokname(st, c)):  P(attach) $(round(get(pa, c, 0.0); digits = 4))",
      isempty(prod_) ? "" : ", product score $(round(prod_[c]; digits = 2))")
    if !isempty(prod_)
      println("       head-property terms: ", join(["$name=$(headval(name, S.head_value(st, t, c, name))) " *
        "$(round(something(findfirst(==(S.head_value(st, t, c, name)), vals) |> j -> j === nothing ? 0.0 : ps[j], 0.0); digits = 3))"
        for (name, (vals, ps)) in props], ", "))
    end
    f === nothing && (println("       (no chain)"); continue)
    println("       prior odds $(round(f.odds[1]; sigdigits = 3))")
    println("       chain steps (each conditioned on all the steps before it; Δ = change in log odds):")
    for (j, x) in enumerate(f.members)
      sl = slotof(c, x)
      dl = log(f.odds[j+1]) - log(f.odds[j])
      println("       $j. ", rpad(sl === nothing ? "?" : slotdesc(st, t, c, sl), 44), " Δ ", lpad((dl >= 0 ? "+" : "") * string(round(dl; digits = 2)), 6),
        "   odds $(round(f.odds[j]; sigdigits = 3)) -> $(round(f.odds[j+1]; sigdigits = 3))")
    end
    inc = [l for l in events[c].left if l[2]]
    println("       chain log odds $(round(log(f.odds[end]); digits = 2)); leftovers tried $(length(events[c].left)), ",
      "included $(length(inc)), total correction $(round(sum(l -> l[4], inc; init = 0.0); digits = 2)); final log odds $(round(f.logodds; digits = 2))")
    for (sl, _, ctxs, corr) in sort(inc; by = l -> -abs(l[4]))
      abs(corr) < 0.005 && continue
      println("         + ", rpad(slotdesc(st, t, c, sl), 36), lpad(round(corr; digits = 3), 8), "   context: ",
        isempty(ctxs) ? "(none: marginal)" : join([slotdesc(st, t, c, q) for q in ctxs if q > 0], ", "))
    end
    nz = count(l -> abs(l[4]) < 0.005, inc)
    nz > 0 && println("         ($nz more leftovers with corrections under 0.005)")
  end
end

"""
The head chosen for t from the chains alone (no leftover corrections): softmax over the candidates'
chain log odds, with domains=product plus the head properties' log probabilities of each candidate's
values.  t's attachment and head properties must be hidden.
"""
function chain_only_choice(st, t)
  toks = st.layers[:token]
  decs = toks.spec.decisions
  cands = decs[1].candidates(st, t)
  lo = Dict{UInt32,Float64}()
  S.CHAIN_TRACE[] = (ev, data) -> ev == :final && (lo[data.cand] = log(data.odds[end]))
  empty!(m.memo)
  try
    decide(m, st, (:token, 1, t))
  finally
    S.CHAIN_TRACE[] = nothing
  end
  isempty(lo) && return PENDING
  mx = maximum(values(lo))
  lz = mx + log(sum(exp(v - mx) for v in values(lo)))
  sc = Dict(c => get(lo, c, -Inf) - lz for c in cands)
  if domains == :product
    for (k, d) in enumerate(decs)
      haskey(S.ALLHEADSETS, d.name) || continue
      dist = S.commit_dist(m, st, (:token, k, t))
      dist === nothing && continue
      vals, lps = dist
      for c in cands
        v = S.head_value(st, t, c, d.name)
        v == PENDING && continue
        j = findfirst(==(v), vals)
        sc[c] += j === nothing ? log(1e-12) : lps[j]
      end
    end
  end
  argmax(sc)
end

mutable struct ExplainRec
  r0::CeilingRecorder
  rgold::CeilingRecorder                           # the same with all head properties given (gold): the head-property ceiling
  shown::Int
end
observe!(::ExplainRec, st, key, choice) = nothing
segment_discard!(::ExplainRec) = nothing
function segment_done!(r::ExplainRec, st)
  toks = st.layers[:token]
  sents = unique(toks[:sentence, t] for t in 1:toks.ncols if toks[:sentence, t] != PENDING)
  for sid in sents
    r.shown >= cfg["nsent"] && return
    ts = sort([t for t in 1:toks.ncols if toks[:sentence, t] == sid]; by = t -> toks[:first_char, t])
    (length(ts) <= cfg["maxlen"] && all(t -> Int(toks[:last_char, t]) <= st.closed_through && toks[:attach, t] != PENDING, ts)) || continue
    res = [(t, toks[:attach, t], first(pipeline_attach!(r.r0, st, t, toks[:attach, t]))) for t in ts]
    wrong = [(t, g, p) for (t, g, p) in res if g != p]
    1 <= length(wrong) <= cfg["maxerr"] || continue
    # only errors the head-property ceiling (gold head properties) gets right
    all(((t, g, p),) -> first(pipeline_attach!(r.rgold, st, t, g)) == g, wrong) || continue
    if cfg["chainonly"] == "true"                     # only errors the chains alone already make
      chainsame = all(wrong) do (t, g, p)
        names = [:attach; S.headnames(toks)]
        old = [toks[n, t] for n in names]
        foreach(n -> (toks[n, t] = PENDING), names)
        S.refresh!(st)
        try
          chain_only_choice(st, t) == p
        finally
          foreach(((n, v),) -> (toks[n, t] = v), zip(names, old))
          S.refresh!(st)
        end
      end
      chainsame || continue
    end
    r.shown += 1
    println("\n================ \"", join([S.tokstr(st, t) for t in ts], " "), "\"")
    for (t, g, p) in res
      println("  ", rpad(S.tokstr(st, t), 14), "gold -> ", rpad(tokname(st, g), 12), g == p ? "ok" : "WRONG: predicted -> $(tokname(st, p))")
    end
    for (t, g, p) in wrong
      println("\n  ---- '$(S.tokstr(st, t))': gold head '$(tokname(st, g))', predicted '$(tokname(st, p))'")
      names = [:attach; S.headnames(toks)]
      old = [toks[n, t] for n in names]
      foreach(n -> (toks[n, t] = PENDING), names)
      S.refresh!(st)
      try
        explain_attach(st, t, g, p)
      finally
        foreach(((n, v),) -> (toks[n, t] = v), zip(names, old))
        S.refresh!(st)
      end
    end
  end
end

println(s.modelname, (domains == :off ? "" : "  [domains=$domains domeps=$domeps]"))
docs = sort([d for (_, d) in s.devdocs]; by = d -> length(d.text))[1:min(cfg["docs"], end)]
rec = ExplainRec(CeilingRecorder(m, first(docs), false, cfg["headconf"]; domains, domeps),
  CeilingRecorder(m, first(docs), false, 1.0, collect(keys(S.ALLHEADSETS))), 0)
for d in docs
  rec.shown >= cfg["nsent"] && break
  rec.r0.doc = d; rec.rgold.doc = d
  process(d, specs, OracleModel(), s.lex; observer = rec)
end
rec.shown == 0 && println("no sentence with 1..$(cfg["maxerr"]) wrong heads and at most $(cfg["maxlen"]) words in these docs")
