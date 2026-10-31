# ceiling.jl  --  perfect-context ceiling for a dev_eval.jl model
#
#   julia --project=experiments experiments/ceiling.jl [key=value ...]     (same options as dev_eval.jl)
#
# Builds the model exactly as dev_eval.jl does, then scores each dev decision with every other
# decision at its gold value (see ceiling_lib.jl).  Tokenization is gold, so:
#   gap_acc    gap decisions right
#   uas        attach right, over gold tokens scored
#   las        attach and deprel both right (each predicted with the other at gold)
#   head_lost  fraction of tokens whose gold head wasn't among the constrained candidates (with
#              head-property decisions: after narrowing by the predicted ones)
#   coverage   fraction of gold tokens scored (the rest stayed pending in the oracle run)
#   tag accuracies (in ceiling_breakdown.csv, dim "tag"): each tag decision with everything else gold;
#              skipped with tags=false, which is much faster (gold tags still fill the context)
#   headconf   pipeline (head-property decisions): commit a predicted head property only if its
#              probability is at least this (0 = always); "tag" rows "<name>@committed" are those
#   headgold   pipeline: these head properties are given at their gold values, not predicted
#              (all, or a list like head_dir,head_dist): a ceiling for the attachment mechanism
#   domains    pipeline with domain variables instead of commits: off | prune | product (src/domains.jl);
#              domeps: the probability below which a value leaves its domain (default 1e-3)
#   prodbeta   domains=product: each head-property term raised to this power (default 1)
#   prodprops  domains=product: only these head properties' terms enter the product (e.g. head_xpos,head_dist;
#              "" = all); the others still narrow the domains
#   dcstats    true: count possible double counting in accum and greedy leftovers -- per leftover taken, the largest |Yule's Q|
#              to a counted feature (chain, earlier leftovers) outside its context; prints the share with |Q| >= 0.5
#              and the |corr|-weighted mean
#   on         dev (default) | tune: score the held-out tuning docs instead (dev_eval.jl tune_sents)
#   calibrate  true: also fit the weight of the greedy leftovers' corrections per decision type
#              (score = chain + alpha * corrections, maximizing the likelihood of the gold choices
#              under choose's softmax), report calibration, and write alpha.tsv (run dir and events
#              dir); use it with leftalpha=<file>.  Normally with on=tune.
#   depfreq, headfreq (breakdown dims): attachment ceiling by how often the dependent's / gold head's
#              form occurs in the training sample (0, 1-4, 5-49, 50+; root heads apart)
#
# Results are appended to experiments/ceiling.csv: one row per treebank plus one for ALL.

include(joinpath(@__DIR__, "dev_eval.jl"))

const CEILING_RESULTS = joinpath(HERE, "ceiling.csv")

const BREAKDOWN_RESULTS = joinpath(HERE, "ceiling_breakdown.csv")

"Attachment ceiling by gold distance and gold deprel, the kinds of attachment error, and tag accuracies (ALL treebanks)."
function write_breakdown(runid, modelname, cfg, c)
  rows = Dict{Tuple{String,String},Dict{String,Int}}()
  for (k, v) in c
    parts = split(String(k), '|')
    length(parts) == 3 || continue
    get!(rows, (parts[1], parts[2]), Dict{String,Int}())[parts[3]] = v
  end
  nerr = sum(m["n"] for ((d, _), m) in rows if d == "err"; init = 0)
  newfile = !isfile(BREAKDOWN_RESULTS)
  open(BREAKDOWN_RESULTS, "a") do io
    newfile && println(io, "run,model,dim,value,n,share,uas,las,note")
    for dim in ("dist", "deprel", "depfreq", "headfreq", "err", "tag")
      sel = sort([(v, m) for ((d, v), m) in rows if d == dim]; by = x -> -x[2]["n"])
      isempty(sel) && continue
      tot = dim == "err" ? nerr : sum(x -> x[2]["n"], sel; init = 0)
      header = dim == "err" ? "" : dim == "tag" ? "acc" : "uas     las"
      println("\n", rpad(dim, 14), rpad("n", 8), rpad("share", 8), header)
      for (v, m) in sel
        n = m["n"]
        # tag rows: uas column holds the tag accuracy
        u, l = dim == "err" ? (NaN, NaN) : dim == "tag" ? (m["ok"] / n, NaN) : (m["uas"] / n, m["las"] / n)
        println(rpad(v, 14), rpad(n, 8), rpad(round(n / tot; digits = 3), 8),
          dim == "err" ? "" : dim == "tag" ? string(round(u; digits = 4)) : string(rpad(round(u; digits = 3), 8), round(l; digits = 3)))
        println(io, join((runid, repr(modelname), dim, v, n, n / tot, u, l, repr(cfg["note"])), ","))
      end
    end
  end
end

function ceiling_main(cfg)
  s = setup(cfg)
  (; tbs, modelname) = s
  devdocs = cfg["on"] == "tune" ? s.tunedocs : cfg["on"] == "dev" ? s.devdocs : error("on: dev or tune")
  cal = cfg["calibrate"] == "true"
  rundir = start_workers(cfg, s)
  Distributed.remotecall_eval(Main, workers(), :(include($(joinpath(HERE, "ceiling_lib.jl")))))
  freq = Dict{String,Int}()                          # training form frequencies, for the depfreq/headfreq dims
  for d in s.trainsample, span in keys(d.gold.head)
    f = lowercase(String(d.text[span]))
    freq[f] = get(freq, f, 0) + 1
  end
  Distributed.remotecall_eval(Main, workers(), :(FORMFREQ[] = $freq))
  pp = isempty(cfg["prodprops"]) ? nothing : Set(Symbol.(split(cfg["prodprops"], ',')))
  pp === nothing || all(n -> haskey(SeqLayers.ALLHEADSETS, n), pp) || error("prodprops: names are $(join(keys(SeqLayers.ALLHEADSETS), ", "))")
  Distributed.remotecall_eval(Main, workers(), :(SeqLayers.PROD_BETA[] = $(cfg["prodbeta"]); SeqLayers.PROD_PROPS[] = $pp))
  cfg["dcstats"] == "true" && Distributed.remotecall_eval(Main, workers(), :(SeqLayers.DC_STATS[] = Dict{Symbol,Float64}()))
  t1 = time()
  order = sortperm(devdocs; by = p -> length(p[2].text), rev = true)
  scoretags = cfg["tags"] == "true"
  headconf = cfg["headconf"]
  headgold = cfg["headgold"] == "all" ? collect(keys(SeqLayers.ALLHEADSETS)) :
    isempty(cfg["headgold"]) ? Symbol[] : Symbol.(split(cfg["headgold"], ','))
  all(n -> haskey(SeqLayers.ALLHEADSETS, n), headgold) || error("headgold: names are $(join(keys(SeqLayers.ALLHEADSETS), ", "))")
  domains, domeps = Symbol(cfg["domains"]), cfg["domeps"]
  res = pmap(p -> (p[1], ceiling_counts(p[2], WSPECS, WMODEL, WLEX; scoretags, headconf, headgold, collect = cal, domains, domeps)), devdocs[order])
  counts = cal ? [(tb, c[1]) for (tb, c) in res] : res
  parts = cal ? reduce(vcat, [c[2] for (_, c) in res]; init = Any[]) : Any[]
  save_countcache(cfg, s)
  score_s = time() - t1
  rmprocs(workers())
  rm(rundir; recursive = true)

  bytb = Dict(tb => Dict{Symbol,Int}() for tb in [tbs; "ALL"])
  for (tb, c) in counts, (k, v) in c
    bytb[tb][k] = get(bytb[tb], k, 0) + v
    bytb["ALL"][k] = get(bytb["ALL"], k, 0) + v
  end
  runid = Dates.format(now(), "yyyymmdd-HHMMSS")
  newfile = !isfile(CEILING_RESULTS)
  open(CEILING_RESULTS, "a") do io
    newfile && println(io, "run,model,train_sents,dev_sents,treebank,words,gap_acc,uas,las,head_lost,coverage,notready,score_s,note")
    # (the pipeline's headconf goes in the note)
    println(rpad("treebank", 12), rpad("words", 8), "gap_acc uas     las     head_lost coverage")
    for tb in [tbs; "ALL"]
      c = bytb[tb]
      n = max(c[:tok_n], 1)
      x = (c[:gap_ok] / max(c[:gap_n], 1), c[:uas] / n, c[:las] / n, c[:head_lost] / n, c[:tok_n] / max(c[:tok_gold], 1))
      println(rpad(tb, 12), rpad(c[:tok_gold], 8), join((rpad(round(v; digits = 4), 8) for v in x), ""))
      println(io, join((runid, repr(modelname), cfg["train_sents"], cfg["dev_sents"], tb, c[:tok_gold], x...,
          c[:notready], round(score_s; digits = 1), repr(cfg["note"])), ","))
    end
  end
  a = bytb["ALL"]
  a[:inferred] + a[:nocands] > 0 && println("\npipeline: $(a[:inferred]) attachments inferred ($(round(a[:inferred] / max(a[:tok_n], 1); digits = 3)) of tokens), ",
    "$(round(a[:inferred_ok] / max(a[:inferred], 1); digits = 3)) of them right; $(a[:nocands]) left with no candidate")
  get(a, :dom_cands, 0) > 0 && println("domains: attachment candidates $(a[:dom_cands]) -> $(a[:dom_alive]) after propagation ",
    "($(round(a[:dom_alive] / a[:dom_cands]; digits = 3)))")
  if get(a, :dc_n, 0) > 0                              # dcstats=true (accum / greedy leftovers; see SeqLayers.dc_record!)
    n, hi, cr, ms = a[:dc_n] / 1000, a[:dc_hi] / 1000, a[:dc_corr] / 1000, a[:dc_mass] / 1000
    println("double counting: $(round(Int, n)) leftovers taken, $(round(hi / n; digits = 3)) with |Q| >= $(SeqLayers.DC_HI) to a counted ",
      "feature outside the context; |corr|-weighted max |Q| $(round(ms / cr; digits = 3)); mean |corr| $(round(cr / n; digits = 3))")
  end
  if get(a, :cs_zero, 0) + get(a, :cs_part, 0) + get(a, :cs_keep, 0) > 0   # chainsel: what happened to the chain steps
    z, p, kp = get(a, :cs_zero, 0) / 1000, get(a, :cs_part, 0) / 1000, get(a, :cs_keep, 0) / 1000
    n = z + p + kp
    println("chainsel: $(round(Int, n)) chain steps -- collapsed to reference $(round(z / n; digits = 3)), pulled partway $(round(p / n; digits = 3)), kept (w >= 0.9) $(round(kp / n; digits = 3))")
  end
  if any(k -> startswith(String(k), "bfs_"), keys(a))  # accumby=bfs: at which backoff level W was first supported
    ks = sort([k for k in keys(a) if startswith(String(k), "bfs_")])
    n = sum(a[k] for k in ks) / 1000
    println("bfs: $(round(Int, n)) searches -- ", join(["$(String(k)[5:end]) $(round(a[k] / 1000 / n; digits = 3))" for k in ks], ", "))
  end
  if get(a, :look_supp, 0) + get(a, :look_dep, 0) > 0   # accumby=look1*: which rule took each backoff step
    sp, dp = get(a, :look_supp, 0) / 1000, get(a, :look_dep, 0) / 1000
    println("look1: $(round(Int, sp + dp)) backoff steps -- to a supported context $(round(sp / (sp + dp); digits = 3)), by dependence $(round(dp / (sp + dp); digits = 3))")
  end
  if get(a, :lw_pair, 0) + get(a, :lw_ctx, 0) > 0     # lookwback: interaction from the context vs the pairwise backoff
    pp, cc = get(a, :lw_pair, 0) / 1000, get(a, :lw_ctx, 0) / 1000
    println("lookwback: $(round(Int, pp + cc)) interactions -- pairwise $(round(pp / (pp + cc); digits = 3)), context (or blended) $(round(cc / (pp + cc); digits = 3))")
  end
  get(a, :pn_restart, 0) > 0 && println("posnegrestart: $(round(Int, a[:pn_restart] / 1000)) chain restarts")
  get(a, :pooled, 0) > 0 && println("explainpool: $(round(Int, a[:pooled] / 1000)) explained-away leftovers joined the context pool")
  if get(a, :tw_w, 0) + get(a, :tw_t, 0) > 0          # accumby=mintw: which proposal each backoff step took
    w, t, wl = a[:tw_w] / 1000, a[:tw_t] / 1000, get(a, :tw_wlost, 0) / 1000
    println("mintw: $(round(Int, w + t)) backoff steps, ΔW taken $(round(w / (w + t); digits = 3)), ΔT taken $(round(t / (w + t); digits = 3)) ",
      "(of which ΔW available but larger: $(round(Int, wl)))",
      haskey(a, :tw_cont) ? "; mintwc: $(round(Int, a[:tw_cont] / 1000)) backoff steps past first support" : "")
  end
  write_breakdown(runid, modelname, cfg, bytb["ALL"])
  cal && calibrate_report(cfg, s, parts)
  println("score $(round(score_s; digits = 1)) s, run $runid → $CEILING_RESULTS, $BREAKDOWN_RESULTS")
end

DEFAULTS["tags"] = "true"
DEFAULTS["headconf"] = 0.9
DEFAULTS["headgold"] = ""
DEFAULTS["domains"] = "off"
DEFAULTS["domeps"] = 1e-3
DEFAULTS["prodbeta"] = 1.0
DEFAULTS["dcstats"] = "false"
DEFAULTS["prodprops"] = ""
DEFAULTS["on"] = "dev"
DEFAULTS["calibrate"] = "false"

"log P(gold) under choose's softmax for scores L + a*C (plus the fixed wait score of open decisions)."
function goldll(L, C, w, g, a)
  s = L .+ a .* C
  isnan(w) || (s = [s; w])
  mx = maximum(s)
  s[g] - mx - log(sum(exp.(s .- mx)))
end

"Confidence of the chosen candidate under choose (with wait), and whether it is gold."
function chosen_conf(L, C, w, g, a)
  s = L .+ a .* C
  b = argmax(s)
  z = isnan(w) ? s : [s; w]
  mx = maximum(z)
  exp(s[b] - mx) / sum(exp.(z .- mx)), b == g
end

"""
Fit alpha per decision type on the collected parts; print the fit and a calibration table (chosen
confidence bins vs accuracy) at alpha = 1 and at the fitted alpha; write alpha.tsv.
"""
function calibrate_report(cfg, s, parts)
  sp = default_specs(wordclasses = s.wc, suffixes = s.sfx, noise = cfg["noise"], rootabsent = cfg["rootabsent"] == "true",
    relations = cfg["relations"] == "true", headsdist = cfg["headsdist"] == "true")
  names = Dict(SeqLayers.typekey(:char, 1) => "gap")
  foreach(((k, d),) -> (names[SeqLayers.typekey(:token, k)] = string(d.name)), enumerate(sp[2].decisions))
  bytk = Dict{UInt64,Vector{Any}}()
  foreach(p -> push!(get!(() -> Any[], bytk, p[1]), p), parts)
  grid = 0.0:0.02:2.0
  bins = (0.5, 0.8, 0.9, 0.95, 0.99, 1.0)
  out = joinpath(s.recipe[1], "alpha.tsv")
  lines = String[]
  println("\ncalibration: weight of the leftovers' corrections per decision type (fit on $(cfg["on"]) docs)")
  println(rpad("decision", 12), lpad("n", 7), lpad("alpha", 7), lpad("ll(1)", 10), lpad("ll(alpha)", 11), lpad("acc(1)", 8), lpad("acc(a)", 8))
  for (tk, ps) in sort(collect(bytk); by = p -> get(names, p[1], ""))
    ll(a) = sum(p -> goldll(p[2], p[3], p[4], p[5], a), ps)
    lls = [ll(a) for a in grid]
    a = grid[argmax(lls)]
    acc(a) = count(p -> chosen_conf(p[2], p[3], p[4], p[5], a)[2], ps) / length(ps)
    nm = get(names, tk, string(tk; base = 16))
    println(rpad(nm, 12), lpad(length(ps), 7), lpad(a, 7), lpad(round(ll(1.0); digits = 1), 10), lpad(round(maximum(lls); digits = 1), 11),
      lpad(round(acc(1.0); digits = 4), 8), lpad(round(acc(a); digits = 4), 8))
    push!(lines, string(tk; base = 16) * "\t" * nm * "\t" * string(a))
    for (lab, aa) in (("alpha=1", 1.0), ("alpha=$a", a))
      cs = [chosen_conf(p[2], p[3], p[4], p[5], aa) for p in ps]
      row = String[]
      lo = 0.0
      for hi in bins
        sel = [c for c in cs if lo <= c[1] < hi || (hi == 1.0 && c[1] == 1.0)]
        isempty(sel) || push!(row, "[$(lo),$(hi)): n $(length(sel)) conf $(round(sum(first, sel) / length(sel); digits = 3)) acc $(round(count(last, sel) / length(sel); digits = 3))")
        lo = hi
      end
      println("    ", rpad(lab, 12), join(row, "; "))
    end
  end
  for f in (out, joinpath(HERE, "runs", "alpha-" * Dates.format(now(), "yyyymmdd-HHMMSS") * ".tsv"))
    open(f, "w") do io
      println(io, "# typekey (hex)\tdecision\talpha  -- ceiling.jl calibrate=true on=$(cfg["on"]), $(s.modelname)")
      foreach(l -> println(io, l), lines)
    end
  end
  println("alpha table -> $out")
end
ceiling_main(parse_cfg(ARGS))
