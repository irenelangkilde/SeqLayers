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
  (; tbs, devdocs, modelname) = s
  rundir = start_workers(cfg, s)
  Distributed.remotecall_eval(Main, workers(), :(include($(joinpath(HERE, "ceiling_lib.jl")))))
  freq = Dict{String,Int}()                          # training form frequencies, for the depfreq/headfreq dims
  for d in s.trainsample, span in keys(d.gold.head)
    f = lowercase(String(d.text[span]))
    freq[f] = get(freq, f, 0) + 1
  end
  Distributed.remotecall_eval(Main, workers(), :(FORMFREQ[] = $freq))
  t1 = time()
  order = sortperm(devdocs; by = p -> length(p[2].text), rev = true)
  scoretags = cfg["tags"] == "true"
  headconf = cfg["headconf"]
  headgold = cfg["headgold"] == "all" ? collect(keys(SeqLayers.ALLHEADSETS)) :
    isempty(cfg["headgold"]) ? Symbol[] : Symbol.(split(cfg["headgold"], ','))
  all(n -> haskey(SeqLayers.ALLHEADSETS, n), headgold) || error("headgold: names are $(join(keys(SeqLayers.ALLHEADSETS), ", "))")
  counts = pmap(p -> (p[1], ceiling_counts(p[2], WSPECS, WMODEL, WLEX; scoretags, headconf, headgold)), devdocs[order])
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
  write_breakdown(runid, modelname, cfg, bytb["ALL"])
  println("score $(round(score_s; digits = 1)) s, run $runid → $CEILING_RESULTS, $BREAKDOWN_RESULTS")
end

DEFAULTS["tags"] = "true"
DEFAULTS["headconf"] = 0.9
DEFAULTS["headgold"] = ""
ceiling_main(parse_cfg(ARGS))
