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
# detail=true: per chain step the feature's prior (marginal) effect, its effect in the chain, and the records behind
#   each; per leftover (in the order taken, accum) the running estimate and the generalization walk: each context
#   tried, its odds and records, the leftover's correction there; and positive vs negative evidence summed apart.
# sent=text: only sentences containing this text.
# contrast=accumby:odds,ctxmode:accum: only errors a second model (these chain options overriding this run's) gets
#   right; each wrong head is then explained under both models.
# deprel=conj: only errors whose token has this gold deprel (the sentence's other errors are still listed).
# goldfix=false: don't require that gold head properties fix the errors.
# singletons=true: instead, over every dev doc, list the features that are the whole of a one-step attachment chain:
#   per feature (as named in the traces), how many candidates' chains, how many of them the gold head, the step's Δ,
#   and how many took leftovers (a nonzero total correction) with their mean total.
# prodprops=head_xpos,head_dir,head_dist: only these head properties' terms in the product (as ceiling.jl).
# nopunct=true: skip errors where the token, its gold head or its predicted head is punctuation (upos PUNCT).
# headblame=true (domains=product): only errors the attachment alone gets right (argmax P(attach) is the gold head) --
#   the head-property terms flip them; per property, the log margin log P(pred's value) - log P(gold's value).
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
DEFAULTS["headblame"] = "false"
DEFAULTS["detail"] = "false"
DEFAULTS["sent"] = ""
DEFAULTS["contrast"] = ""
DEFAULTS["nopunct"] = "false"
DEFAULTS["prodprops"] = ""
DEFAULTS["deprel"] = ""
DEFAULTS["goldfix"] = "true"
DEFAULTS["singletons"] = "false"
DEFAULTS["headconf"] = 1.0
DEFAULTS["domains"] = "off"
DEFAULTS["domeps"] = 1e-3
cfg = parse_cfg(["model=chain"; ARGS])
s = setup(cfg)
idx = s.idx
widx = CharIndex(idx.fwd, idx.rev, idx.gap, idx.saL, idx.saR, idx.cumL, idx.cumR, idx.prior, IdDict{Doc,Int}(), idx.maxlen, idx.mincount, idx.beta)
specs = default_specs(charindex = widx, wordclasses = s.wc, suffixes = s.sfx, noise = cfg["noise"], rootabsent = cfg["rootabsent"] == "true",
  projective = cfg["projective"] == "true", relations = cfg["relations"] == "true", headsdist = cfg["headsdist"] == "true", relbounded = cfg["relbounded"] == "true", relnone = cfg["relnone"] == "true", treerank = cfg["treerank"] == "true", headtrank = cfg["headtrank"] == "true", headwc = isempty(cfg["headwc"]) ? Int[] : parse.(Int, split(cfg["headwc"], ',')), headwcclasses = S.HEADWC[], headsibs = headsib_names(cfg), idxeq = cfg["idxeq"] == "true",
  kinitems = cfg["kinitems"] == "true", headkin = headkin_names(cfg), neighbor2 = cfg["neighbor2"] == "true",
  oclass = cfg["oclass"] == "true", bioes = cfg["bioes"] == "true", fwdsets = cfg["fwdsets"] == "true", shapeitems = cfg["shapeitems"] != "false")
eventsdir, opts = s.recipe
pc = S.load_counts(eventsdir, eventsig(eventsdir))
function build_model(opts)
  m = ChainModel(eventsdir; opts...)
  pc !== nothing && (m.pcounts = pc)
  (cfg["poolsigma"] > 0 || get(opts, :ctxmode, :greedy) != :greedy) && (m.generalize = S.generalization_slots(specs))
  cfg["classchain"] == "true" || (m.nochain = S.wclass_slots(specs))
  cfg["ctxscope"] == "heads" && (m.ctxtypes = Set(S.typekey(:token, k) for (k, d) in enumerate(specs[2].decisions) if k == 1 || haskey(S.ALLHEADSETS, d.name)))
  m
end
optval(v) = v in ("true", "false") ? v == "true" : (x = tryparse(Int, v)) !== nothing ? x : (x = tryparse(Float64, v)) !== nothing ? x : Symbol(v)
m = build_model(opts)
contrast_opts = isempty(cfg["contrast"]) ? nothing :
  merge(opts, NamedTuple(Symbol(k) => optval(v) for (k, v) in (split(kv, ':') for kv in split(cfg["contrast"], ','))))
m2 = contrast_opts === nothing ? nothing : build_model(contrast_opts)
mainmodel = m
domains, domeps = Symbol(cfg["domains"]), cfg["domeps"]
isempty(cfg["prodprops"]) || (S.PROD_PROPS[] = Set(Symbol.(split(cfg["prodprops"], ','))))   # (as ceiling.jl)

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
  f == :oclass && return S.OCLASS_LABELS[v]
  string(v)
end
const BIOES_NAMES = ("S", "B", "E", "I", "edge")
bioesname(v) = v == PENDING ? "?" : v <= 5 ? BIOES_NAMES[v] : "root"
bioespair(v) = v == PENDING ? "?" : bioesname(div(v - 1, 5) + 1) * "|" * bioesname(mod(v - 1, 5) + 1)
kinname(v) = v == PENDING ? "?" : S.KIN_LABELS[v]
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
  if a > length(lp)                                # the items appended after the link paths, in specs order
    k = sl - 2 - 2 * length(lp)
    if cfg["relations"] == "true"                  # hypothetical siblings under c (14 items)
      if k <= 14
        f = S.RelFlags(cfg["relbounded"] == "true", cfg["relnone"] == "true")
        kind, field, v = S.hypothetical_sibling_values(st, t, c, f)[k]
        kn = replace(String(kind), "_sibling" => "")
        val = v == PENDING ? "?" : v == NONE ? "none" : field == :upos ? S.TAGSETS[:upos][v] : field == :deprel ? S.DEPRELS[v] :
          field == :dir ? (v == 1 ? "left" : "right") : ("1", "2", "3-5", "6-10", "11+")[v]
        return "hypsib.$kn.$field=$val"
      end
      k -= 14
    end
    if cfg["treerank"] == "true"
      k == 1 && return "cand.treerank=" * (c == NONE ? "root" : string(S.tree_rank(st, t, c)))
      k -= 1
    end
    if cfg["idxeq"] == "true"
      if k <= 6
        names = ("tok.right_neighbor=cand.depleft1", "tok.left_neighbor=cand.depright1", "tok.left_neighbor.head=cand",
          "tok.right_neighbor.head=cand", "cand.head=tok.left_neighbor", "cand.head=tok.right_neighbor")
        v = S.idxeq_values(st, t, c)[k]
        return "idxeq:" * names[k] * "?" * (v == PENDING ? "unknown" : v == 1 ? "yes" : "no")
      end
      k -= 6
    end
    if cfg["kinitems"] == "true"                   # the token's kinship to its left / right neighbour under c
      k <= 2 && return "kin.$(k == 1 ? "left" : "right")=" * kinname(S.kin_value(st, t, c, k == 1 ? -1 : 1))
      k -= 2
    end
    if cfg["bioes"] == "true"                      # subtree labels: left gap pair, right gap pair, the candidate's
      if k <= 3
        v = S.bioes_values(st, t, c)[k]
        return k == 1 ? "bioes.leftgap(left|tok)=" * bioespair(v) : k == 2 ? "bioes.rightgap(tok|right)=" * bioespair(v) :
          "bioes.cand=" * bioesname(v)
      end
      k -= 3
    end
    if cfg["fwdsets"] == "true"                    # forward-checking sets, in fwd_set_items' order
      sets = Any[]
      cfg["kinitems"] == "true" && append!(sets, [("set.kin.left", S.kin_set(st, t, c, -1), kinname), ("set.kin.right", S.kin_set(st, t, c, 1), kinname)])
      cfg["bioes"] == "true" && append!(sets, [("set.bioes.leftgap", S.bioes_gap_set(st, t, c, -1), bioespair), ("set.bioes.rightgap", S.bioes_gap_set(st, t, c, 1), bioespair)])
      cfg["idxeq"] == "true" && append!(sets, [("set.idxeq$j", S.idxeq_set(st, t, c, j), v -> v == 1 ? "yes" : "no") for j in 3:6])
      if k <= length(sets)
        nm, vs, f = sets[k]
        return nm * "=" * (vs === nothing ? "?" : "{" * join(f.(vs), ",") * "}")
      end
      k -= length(sets)
    end
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
  println("    head properties predicted (with the attachment and all of them hidden; values are probabilities):")
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
  S.CHAIN_TRACE[] = (ev, data) -> ev == :final ? (events[data.cand] = (left = events[data.cand].left, final = data)) :
    ev == :reset ? (empty!(events[data.cand].left); empty!(accum[data.cand]); nothing) : nothing
  S.LEFTOVER_TRACE[] = (sl, inc, ctxs, corr) -> push!(events[S.CHAIN_TRACE_CAND[]].left, (sl, inc, ctxs, corr))
  accum = Dict{Any,Vector{Any}}(c => Any[] for c in cands)
  S.ACCUM_TRACE[] = (sx, T, tries, k, corr, hist) -> push!(accum[S.CHAIN_TRACE_CAND[]], (sx = sx, T = T, tries = tries, corr = corr, hist = hist))
  mdev = Dict{Any,Dict{UInt64,Any}}(c => Dict{UInt64,Any}() for c in cands)      # greedyby=mindev: each leftover's last path
  S.MINDEV_TRACE[] = (X, path, status, corr) -> (mdev[S.CHAIN_TRACE_CAND[]][X] = (path = copy(path), status = status, corr = corr))
  empty!(m.memo)
  try
    decide(m, st, key)
  finally
    S.CHAIN_TRACE[] = nothing; S.LEFTOVER_TRACE[] = nothing; S.ACCUM_TRACE[] = nothing; S.MINDEV_TRACE[] = nothing
  end
  dist = S.commit_dist(m, st, key)
  pa = dist === nothing ? Dict{UInt32,Float64}() : Dict(zip(dist[1], exp.(dist[2])))
  # the product score (domains=product, no pruning: domeps applies as in the run)
  prod_ = Dict{UInt32,Float64}()
  if domains == :product && dist !== nothing
    for c in cands
      sc = log(max(get(pa, c, 0.0), 1e-12))
      for (name, (vals, ps)) in props
        S.PROD_PROPS[] === nothing || name in S.PROD_PROPS[] || continue
        v = S.head_value(st, t, c, name)
        v == PENDING && continue
        k = findfirst(==(v), vals)
        sc += log(max(k === nothing ? 0.0 : ps[k], 1e-12))
      end
      prod_[c] = sc
    end
  end
  if !isempty(prod_)                                 # which head-property terms decide gold vs predicted
    pv(name, c) = ((vals, ps) = props[name]; j = findfirst(==(S.head_value(st, t, c, name)), vals); j === nothing ? 1e-12 : max(ps[j], 1e-12))
    margins = sort([(name, log(pv(name, pred)) - log(pv(name, gold))) for name in keys(props)]; by = x -> -x[2])
    println("    log margins predicted - gold (log of probability ratios; > 0 favours the predicted head):  attach ", fmtd(log(max(get(pa, pred, 0.0), 1e-12)) - log(max(get(pa, gold, 0.0), 1e-12))),
      ";  ", join(["$name $(fmtd(d))" for (name, d) in margins if S.PROD_PROPS[] === nothing || name in S.PROD_PROPS[]], ", "),
      ";  total ", haskey(prod_, pred) && haskey(prod_, gold) ? fmtd(prod_[pred] - prod_[gold]) :
        "n/a (predicted head $(tokname(st, pred)) is not among the $(length(cands)) candidates here)")
  end
  score(c) = isempty(prod_) ? get(pa, c, 0.0) : get(prod_, c, -Inf)
  others = [c for c in sort(cands; by = c -> -score(c)) if c != gold && c != pred]
  show = filter(in(cands), unique([gold; pred; others]))[1:min(end, max(2, cfg["explaintop"]))]
  println("    candidates: $(length(cands))", isempty(prod_) ? "" : "   (product score = log P(attach) + Σ log P(head property value): a log probability)")
  for c in show
    f = events[c].final
    tag = c == gold ? "GOLD" : c == pred ? "PREDICTED" : "other"
    println("\n    -- $tag -> $(tokname(st, c)):  P(attach) $(round(get(pa, c, 0.0); digits = 4)) (probability)",
      isempty(prod_) ? "" : ", product score $(round(prod_[c]; digits = 2))")
    if !isempty(prod_)
      println("       head-property terms (probabilities): ", join(["$name=$(headval(name, S.head_value(st, t, c, name))) " *
        "$(round(something(findfirst(==(S.head_value(st, t, c, name)), vals) |> j -> j === nothing ? 0.0 : ps[j], 0.0); digits = 3))"
        for (name, (vals, ps)) in props], ", "))
    end
    f === nothing && (println("       (no chain)"); continue)
    if cfg["detail"] == "true"
      detail(st, t, c, f, items[findfirst(==(c), cands)], accum[c])
      continue
    end
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
    if !isempty(mdev[c])                             # greedyby=mindev: every leftover's path B (a, diff after adding a)
      Ic = items[findfirst(==(c), cands)]
      nm(x) = (sl = findfirst(==(x), Ic); sl === nothing ? "?" : slotdesc(st, t, c, sl))
      println("       mindev paths (diff = log o(x|e,B) - log o(x|B) after each a joins B; [n B / n B∪e], n = chosen/not-chosen):")
      for (X, r) in sort(collect(mdev[c]); by = p -> (p[2].status != :kept, -abs(p[2].corr)))
        println("         ", rpad(string(r.status), 10), rpad(nm(X), 36), lpad(fmtd(r.corr), 7), "   ",
          isempty(r.path) ? "(no supported context)" : join(["$(nm(a)) $(fmtd(d)) [$(fmtn(nb)) / $(fmtn(ne))]" for (a, d, nb, ne) in r.path], "  ->  "))
      end
    end
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

fmtn(n) = n === nothing ? "-" : "$(round(Int, n[1]))/$(round(Int, n[2]))"
fmtd(x) = (x >= 0 ? "+" : "") * string(round(x; digits = 2))

"The detailed view of candidate c's score (detail=true): see the header."
function detail(st, t, c, f, Ic, acc)
  tk = S.typekey(:token, 1)
  base = get(m.typerecs, tk, Int32[])
  p0 = f.odds[1]
  name(x) = (sl = findfirst(==(x), Ic); sl === nothing ? "?" : slotdesc(st, t, c, sl))
  function marginal(x)                              # x alone: its odds from the prior, and its records (chosen / not)
    S0 = S.ChainState(p0, base)
    e = only(S.extend_all(m, S0, 0, [x]))
    y, ok = S.update_odds(m, e, UInt32(0), p0)
    (ok ? log(y) - log(p0) : NaN, get(e[2], UInt32(0), (0.0, 0.0)))
  end
  pos, neg = 0.0, 0.0
  println("       prior: log odds $(fmtd(log(p0))) = odds $(round(p0; sigdigits = 3)) = probability $(round(p0 / (1 + p0); sigdigits = 3))")
  println("       (n = training records: chosen / not chosen.  Δ = change in log odds; e^Δ is the factor on the odds.")
  println("        Leftovers: 'context log odds' is log o(x | context); 'with X' counts the context's records that have the")
  println("        leftover; corr = the leftover's Δ log odds in that context, added to the running estimate.\n" *
    "        'by round': the leftover's corr in each greedy round before it was taken, in that round's context.)")
  println("       CHAIN", " "^41, "Δ alone (vs prior)       Δ in the chain (vs prefix)")
  Sc = S.ChainState(p0, base)
  for (j, x) in enumerate(f.members)
    e = only(S.extend_all(m, Sc, S.depth(Sc), [x]))
    dl = log(f.odds[j+1]) - log(f.odds[j])
    mΔ, mn = marginal(x)
    println("       $j. ", rpad(name(x), 40), lpad(fmtd(mΔ), 7), "  n ", rpad(fmtn(mn), 14), lpad(fmtd(dl), 7), "  n ", fmtn(get(e[2], UInt32(0), (0.0, 0.0))),
      sign(mΔ) != sign(dl) && abs(dl) > 0.2 ? "   <- sign flips" : "")
    dl >= 0 ? (pos += dl) : (neg += dl)
    S.advance!(m, Sc, x, e, f.odds[j+1])
  end
  println("       chain: log odds $(fmtd(log(f.odds[end]))) = odds $(round(f.odds[end]; sigdigits = 3))")
  if isempty(acc)
    println("       (no accum leftovers traced)")
  end
  for (r, a) in enumerate(acc)
    X = Ic[a.sx]
    mΔ, mn = marginal(X)
    println("\n       LEFTOVER $r: ", name(X), "   Δ alone ", fmtd(mΔ), " (n ", fmtn(mn), ");  running estimate before: log odds ", fmtd(a.T))
    prev = nothing
    taken = false
    for (i, tr) in enumerate(a.tries)
      ctxn = [name(x) for x in tr.ctx]
      moves = prev === nothing ? "" : "   [" * join([["-" * x for x in setdiff(prev, ctxn)]; ["+" * x for x in setdiff(ctxn, prev)]], ", ") * "]"
      mark = ""
      if haskey(tr, :how) && tr.how !== nothing               # accumby=mintw(c): which proposal took this step
        h = tr.how
        mark = h.by == :cont ? "  {past support: ΔW $(fmtd(h.dW)) vs first supported}" :
          "  {$(h.by == :W ? "ΔW" : "ΔT") step: ΔT $(round(h.dT; digits = 2)), ΔW $(h.dW === nothing ? "n/a" : round(h.dW; digits = 2))}"
      end
      final = cfg["accumby"] == "mintwc" ? i == length(a.tries) : (!taken && tr.corr !== nothing)
      final && (taken = true; mark *= "  <= TAKEN")
      println("         ctx $i: context log odds ", lpad(fmtd(log(tr.odds)), 6), "  n ", rpad(fmtn(tr.n), 12), " with X: n ", rpad(fmtn(tr.nX), 10),
        "  corr ", tr.corr === nothing ? "unsupported" : fmtd(tr.corr), mark, moves)
      i == 1 && println("                 context: ", join(ctxn, ", "))
      prev = ctxn
    end
    # its score in every greedy round, the context then (the round's first supported one, growing as leftovers join)
    if length(a.hist) > 1
      fc(c) = c === nothing ? "unsup" : fmtd(c)
      println("         by round: [", join([fc(h.corr) for h in a.hist], ", "), "]")
      prev = nothing
      for h in a.hist
        ctxn = [name(x) for x in h.ctx]
        moves = prev === nothing ? "context: " * join(ctxn, ", ") :
          "[" * join([["-" * x for x in setdiff(prev, ctxn)]; ["+" * x for x in setdiff(ctxn, prev)]], ", ") * "]"
        println("           round ", rpad(h.round, 3), " estimate ", lpad(fmtd(h.T), 6), "  context log odds ", lpad(fmtd(log(h.odds)), 6),
          "  with X: n ", rpad(fmtn(h.nX), 10), "  corr ", lpad(fc(h.corr), 6), h.corr === nothing || h.score ≈ h.corr ? "" : "  (weighted " * fmtd(h.score) * ")", "   ", moves)
        prev = ctxn
      end
    end
    a.corr >= 0 ? (pos += a.corr) : (neg += a.corr)
  end
  println("\n       evidence (sums of Δ log odds): positive ", fmtd(pos), ", negative ", fmtd(neg), ";  final log odds ", fmtd(f.logodds),
    " = odds ", round(exp(f.logodds); sigdigits = 3))
end

mutable struct ExplainRec
  r0::CeilingRecorder
  rgold::CeilingRecorder                           # the same with all head properties given (gold): the head-property ceiling
  r2::Union{Nothing,CeilingRecorder}               # the contrast model (contrast=)
  shown::Int
end
ispunct_tok(st, t) = t != NONE && S.TAGSETS[:upos][st.layers[:token][:upos, Int(t)]] == "PUNCT"
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
    cfg["nopunct"] == "true" && (wrong = [(t, g, p) for (t, g, p) in wrong if !(ispunct_tok(st, t) || ispunct_tok(st, g) || ispunct_tok(st, p))])
    isempty(cfg["deprel"]) || (wrong = [(t, g, p) for (t, g, p) in wrong if S.DEPRELS[toks[:deprel, t]] == cfg["deprel"]])
    1 <= length(wrong) <= cfg["maxerr"] || continue
    isempty(cfg["sent"]) || occursin(cfg["sent"], join([S.tokstr(st, t) for t in ts], " ")) || continue
    # only errors the head-property ceiling (gold head properties) gets right
    cfg["goldfix"] == "true" && !all(((t, g, p),) -> first(pipeline_attach!(r.rgold, st, t, g)) == g, wrong) && continue
    r.r2 === nothing || all(((t, g, p),) -> first(pipeline_attach!(r.r2, st, t, g)) == g, wrong) || continue
    if cfg["headblame"] == "true"                     # only errors the attachment alone gets right
      blamed = all(wrong) do (t, g, p)
        names = [:attach; S.headnames(toks)]
        old = [toks[n, t] for n in names]
        foreach(n -> (toks[n, t] = PENDING), names)
        S.refresh!(st)
        try
          empty!(m.memo)
          dist = S.commit_dist(m, st, (:token, 1, t))
          dist !== nothing && dist[1][argmax(dist[2])] == g
        finally
          foreach(((n, v),) -> (toks[n, t] = v), zip(names, old))
          S.refresh!(st)
        end
      end
      blamed || continue
    end
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
        if m2 === nothing
          explain_attach(st, t, g, p)
        else
          println("\n  ==== THIS MODEL (wrong)")
          global m = mainmodel; empty!(m.memo)
          explain_attach(st, t, g, p)
          println("\n  ==== CONTRAST MODEL ($(cfg["contrast"]); right)")
          global m = m2; empty!(m.memo)
          explain_attach(st, t, g, p)
          global m = mainmodel
        end
      finally
        foreach(((n, v),) -> (toks[n, t] = v), zip(names, old))
        S.refresh!(st)
      end
    end
  end
end

println(s.modelname, (domains == :off ? "" : "  [domains=$domains domeps=$domeps]"))
if cfg["singletons"] == "true"
  # every dev doc through the ceiling; each one-step attachment chain noted by its feature's name (value included)
  single = Dict{String,Vector{Any}}()                 # name -> [chains, gold-head ones, sum of Δ, with leftovers, sum of their Δ]
  goldof = Dict{Tuple{Doc,UnitRange{Int}},UnitRange{Int}}()
  curdoc = Ref{Doc}(first(s.devdocs)[2])
  S.SINGLE_HOOK[] = function (st, key, c, Ic, ch, lo)
    S.decisiondef(st, key).name == :attach || return
    t = key[3]
    sl = findfirst(==(only(ch.members)), Ic)
    nm = sl === nothing ? "?" : slotdesc(st, t, c, sl)
    g = get(curdoc[].gold.head, S.tokspan(st, t), nothing)
    isgold = g !== nothing && (c == NONE ? isempty(g) : S.tokspan(st, Int(c)) == g)
    v = get!(single, nm, Any[0, 0, 0.0, 0, 0.0])
    v[1] += 1; v[2] += isgold; v[3] += log(ch.odds[end]) - log(ch.odds[1])
    lc = lo - log(ch.odds[end])                       # the leftovers' total correction
    abs(lc) > 1e-9 && (v[4] += 1; v[5] += lc)
  end
  r = CeilingRecorder(m, first(s.devdocs)[2], false, cfg["headconf"]; domains, domeps)
  t0 = time()
  for (_, d) in s.devdocs
    curdoc[] = d; r.doc = d
    process(d, specs, OracleModel(), s.lex; observer = r)
  end
  S.SINGLE_HOOK[] = nothing
  tot = sum(v[1] for v in values(single); init = 0)
  println("singleton chains: $tot one-step attachment chains, $(length(single)) distinct features ($(round(time() - t0; digits = 1)) s)")
  nl = sum(v[4] for v in values(single); init = 0)
  println("with leftovers (a nonzero total correction): $nl")
  println(rpad("feature", 50), rpad("chains", 8), rpad("gold head", 11), rpad("mean Δ", 9), rpad("w/ leftovers", 14), "their mean Δ")
  for (nm, v) in sort(collect(single); by = p -> -p[2][1])
    println(rpad(nm, 50), rpad(v[1], 8), rpad(v[2], 11), rpad(round(v[3] / v[1]; digits = 2), 9), rpad(v[4], 14),
      v[4] == 0 ? "" : round(v[5] / v[4]; digits = 2))
  end
  exit()
end
docs = sort([d for (_, d) in s.devdocs]; by = d -> length(d.text))[1:min(cfg["docs"], end)]
rec = ExplainRec(CeilingRecorder(m, first(docs), false, cfg["headconf"]; domains, domeps),
  CeilingRecorder(m, first(docs), false, 1.0, collect(keys(S.ALLHEADSETS))),
  m2 === nothing ? nothing : CeilingRecorder(m2, first(docs), false, cfg["headconf"]; domains, domeps), 0)
for d in docs
  rec.shown >= cfg["nsent"] && break
  rec.r0.doc = d; rec.rgold.doc = d; rec.r2 === nothing || (rec.r2.doc = d)
  process(d, specs, OracleModel(), s.lex; observer = rec)
end
rec.shown == 0 && println("no sentence with 1..$(cfg["maxerr"]) wrong heads and at most $(cfg["maxlen"]) words in these docs")
