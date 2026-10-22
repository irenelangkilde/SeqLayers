# dev_eval.jl  --  train on a training sample, evaluate on the UD dev files, log the results
#
#   julia --project=experiments experiments/dev_eval.jl [key=value ...]
#
# Options (defaults in DEFAULTS):
#   model        lattice | chain | count
#   order        lattice max order
#   minsupport   lattice minimum support (Phase N: TUPLE_COUNT_THRESHOLD)
#   diff, wmo    lattice significance thresholds (Phase N: DIFF_THRESHOLD, WMO_THRESHOLD)
#   select       chain criterion: maxdev | avglogodds | clr_mean_abs | clr_rms
#   tau          chain stopping threshold
#   mincount     chain minimum joint records
#   jointmin     chain joint tables for items with at least this many records (0 = off); exact,
#                only changes speed, so it isn't part of the model name
#   screen       chain build-time screen (needs jointmin > 0): rare items, and frequent items whose
#                own and pairwise corrections (λ) are all below this, are treated as absent (0 = off)
#   screenby     how a labeled decision's per-label λ are combined: maxdev | clr_rms
#   rings        chain climbing: directory of word rings (juliaNLP/clusters/rings; "" = off)
#   climb        which attachment slots climb the rings: cand | dep | cand,dep
#   climbleft    a climbing slot as a leftover: deepest | specific | none (see chain.jl)
#   sample       chain sampling constant c (0 = exact counts): scans read at most c/p₂ records,
#                p₂ = probability of the second-ranked candidate; approximate, so in the model name
#   leftovers    chain leftovers: none | marginal | backoff | backoff1 | greedy
#   minctx       greedy leftovers: a leftover needs a context of at least this many features
#   greedyby     greedy leftovers' context choice: correction | odds (see chain.jl)
#   maxctx       greedy leftovers: largest context (default: no limit)
#   relevance    greedy leftovers: skip a leftover whose relevance (max |λ|) is below this (0 = off)
#   poolm        chain candidate pool size m (0 = off; see chain.jl)
#   poollam      candidate pool: a slot pair needs a mean |λ| of at least this (0 = no λ filter)
#   poolgain     candidate pool: a slot pair needs a mean total effect this much above its stronger slot's
#   pairscore    candidate pool: a pair's score: total | min (least of its and its slots' effects) |
#                increment (its excess over its stronger slot)
#   pairfilter   candidate pool: a supported pair removes its slots from the step: true | false
#   keepfrac     chain record lists (exact; speed/memory only, not in the model name)
#   maxcached    chain cache size in record numbers (exact; speed/memory only)
#   group        chain: one record per labeled decision instead of per label: true | false (exact)
#   countcache   chain: reuse and extend the events' saved feature-set counts (counts_*.bin in the
#                events directory, memory-mapped and shared by the workers; not with climbing):
#                true | false (exact)
#   countmin     chain: save only feature sets whose counting cost at least this many records
#   wclasses     word-class decisions from this dir's wordclasses.tsv (juliaNLP/clusters/classes5;
#                see src/wordclass.jl); "" = none.  Events are recorded separately for each dir.
#   classleft    with wclasses: may class slots be leftovers (else chains only): true | false
#   suffixes     suffix features of these lengths, e.g. 3 or 2,3 ("" = none); events recorded apart
#   noise        a random feature with this many values (0 = none; for testing gates); events apart
#   headsdist    a fifth head-property decision, the head's signed distance (L1..L11+, R1..R11+, root):
#                true | false; events apart
#   relations    sibling and child relations (upos, deprel, distance, direction) in the context and
#                attachment features, and each candidate's hypothetical siblings: true | false; events apart
#   projective   attachment candidates whose arc would cross a committed arc are dropped: true | false
#                (decoding only: events are shared with non-projective runs)
#   rootabsent   the root candidate's candidate-side attachment items are absent: true | false;
#                events apart
#   redundant    greedy leftovers: redundant features (supported at the chain's final prefix, change
#                < tau) join the conditioning pool, without a correction of their own: true | false
#   ctxsize      greedy leftovers' context size: min (at least minctx) | chain (exactly the chain's length) |
#                adaptive (at least max(1, min(minctx, chain length + redundant pool)))
#   stepscore    maxdev chain step score: abs (|log y - log x|) | signed (log y - log x: the feature
#                that raises the odds most) | maxmin (smallest |log y - log o| over the chain's odds so far) |
#                maxmin2 (the same with the counts smoothed toward each o in turn)
#   lockstep     maxdev chains: none (each candidate's chain runs to the end) | top2 (only the two
#                leading candidates advance; see run_top2 in chain.jl)
#   pickfewest   chain steps: of the k best-scoring features reaching tau, add the one with the fewest
#                records (0 = off: the best-scoring one)
#   contest      re-decide the top two candidates by a contest chain on their odds ratio, restarting
#                when one drops below a third (maxrestarts): true | false
#   gate         chain relevance gates: admit a slot only where its effect reaches this (0 = off;
#                see Relevance gates in chain.jl)
#   target       chain target mode: true | false
#   maxrestarts  chain target-mode restarts
#   restart      chain target to restart with: immediate | exhaust (maxdev only)
#   events       how lattice/chain training events are recorded: gold (every decision from the
#                finished gold state, omnidirectional) | replay (left-to-right gold replay)
#   train_sents  training sentences per treebank used for counts
#   dev_sents    dev sentences per treebank (0 = all); the first N, so every run sees the same ones
#   workers      parallel decoding processes
#   note         free text saved with the results
#
# Treebanks are the active (uncommented) lines of the Prague training_corpus.txt; each one's
# dev file is the train file with -ud-train replaced by -ud-dev.  The suffix-array index is
# built from all of their training text.  Recorded lattice events are cached under
# experiments/work/ and re-recorded automatically when src/ changes.
#
# Results are appended to experiments/results.csv: one row per treebank plus one for ALL.

using Distributed, Serialization, Dates
using SeqLayers

const HERE = @__DIR__
const CORPUS_LIST = "/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/word-embeddings_3/data/input/prague/training_corpus.txt"
const RESULTS = joinpath(HERE, "results.csv")

const DEFAULTS = Dict{String,Any}("model" => "lattice", "order" => 3, "minsupport" => 2, "diff" => 3.0,
  "wmo" => 1e-3, "train_sents" => 500, "dev_sents" => 300, "workers" => 8, "note" => "",
  "select" => "clr_rms", "tau" => 0.1, "mincount" => 5, "jointmin" => 0, "sample" => 0.0, "rings" => "", "climb" => "cand", "climbleft" => "deepest", "minctx" => 1, "greedyby" => "correction", "maxctx" => 0, "relevance" => 0.0,
  "poolm" => 0, "poollam" => 0.0, "poolgain" => 0.1, "pairscore" => "total", "pairfilter" => "true", "keepfrac" => 1.0, "maxcached" => 200_000_000, "group" => "true", "countcache" => "true", "countmin" => 10_000, "wclasses" => "", "classleft" => "true", "suffixes" => "", "noise" => 0, "rootabsent" => "false", "projective" => "false", "relations" => "false", "headsdist" => "false", "gate" => 0.0, "redundant" => "false", "ctxsize" => "min", "stepscore" => "abs", "lockstep" => "none", "contest" => "false", "pickfewest" => 0, "screen" => 0.0, "screenby" => "maxdev", "leftovers" => "backoff", "target" => "false",
  "maxrestarts" => 2, "events" => "gold", "restart" => "immediate")

function parse_cfg(args)
  cfg = copy(DEFAULTS)
  for a in args
    k, v = split(a, '='; limit = 2)
    haskey(cfg, k) || error("unknown option $k; options: $(join(sort(collect(keys(cfg))), ", "))")
    cfg[k] = cfg[k] isa Int ? parse(Int, v) : cfg[k] isa Float64 ? parse(Float64, v) : String(v)
  end
  cfg
end

tbname(f) = match(r"UD_English-([^/]+)", f)[1]
nsents(d::Doc) = length(d.gold.sentends)

"The first documents of `docs` covering at least n sentences (all of them if n == 0)."
function firstsents(docs, n)
  n == 0 && return docs
  out, k = Doc[], 0
  for d in docs
    k >= n && break
    push!(out, d)
    k += nsents(d)
  end
  out
end

"""
Hash of the package source and the training files' sizes and modification times, so cached
events are re-recorded when either the code or the data changes.
"""
srcstamp() = string(hash(([read(f, String) for f in sort(readdir(joinpath(HERE, "..", "src"); join = true))],
    [(f, filesize(f), mtime(f)) for f in filter(l -> !startswith(l, '#') && !isempty(strip(l)), readlines(CORPUS_LIST))])); base = 16)

"The event files' sizes and modification times: saved counts are only valid for these events."
eventsig(eventsdir) = string(Tuple((filesize(f), mtime(f)) for f in joinpath.(eventsdir, ("cands.bin", "items.bin"))))

"""
Merge the workers' newly computed feature-set counts into the events' count cache (written
atomically); see Persistent counts in chain.jl.
"""
function save_countcache(cfg, s)
  (cfg["countcache"] == "true" && s.recipe !== nothing && isempty(s.recipe[3])) || return
  new = Dict{UInt64,SeqLayers.LabelCounts}()
  for w in workers()
    merge!(new, remotecall_fetch(() -> Main.WMODEL.newcounts, w))
  end
  dir, sig = s.recipe[1], eventsig(s.recipe[1])
  println("count cache: $(length(new)) new feature sets", merge_countcache(dir, sig, new))
end

"Merge `new` counts into the saved ones in `dir` and save; returns a note for the log."
function merge_countcache(dir, sig, new)
  isempty(new) && return ""
  old = SeqLayers.load_counts(dir, sig)
  old !== nothing && filter!(p -> SeqLayers.pcget(old, p.first) === nothing, new)
  packed = SeqLayers.packcounts(new)
  counts = old === nothing ? packed : SeqLayers.mergecounts(old, packed)
  SeqLayers.save_counts(dir, sig, counts)
  ", $(length(counts)) in all → $dir"
end

"Load the treebanks and build (or load cached events for) the model.  Also used by ceiling.jl."
function setup(cfg)
  trainfiles = filter(l -> !startswith(l, '#') && !isempty(strip(l)), readlines(CORPUS_LIST))
  devfiles = replace.(trainfiles, "-ud-train" => "-ud-dev")
  tbs = tbname.(trainfiles)
  println("treebanks: ", join(tbs, ", "))

  t0 = time()
  alltrain = [read_conllu(f) for f in trainfiles]
  idx = CharIndex(reduce(vcat, alltrain))
  wc = isempty(cfg["wclasses"]) ? nothing : load_wordclasses(cfg["wclasses"])
  sfx = isempty(cfg["suffixes"]) ? Int[] : parse.(Int, split(cfg["suffixes"], ','))
  specs = default_specs(charindex = idx, wordclasses = wc, suffixes = sfx, noise = cfg["noise"], rootabsent = cfg["rootabsent"] == "true",
    projective = cfg["projective"] == "true", relations = cfg["relations"] == "true",
    headsdist = cfg["headsdist"] == "true")
  # same Doc objects as in the index, so their suffix-array features are leave-one-out
  trainsample = reduce(vcat, [firstsents(ds, cfg["train_sents"]) for ds in alltrain])
  devdocs = [(tb, d) for (tb, f) in zip(tbs, devfiles) for d in firstsents(read_conllu(f), cfg["dev_sents"])]
  println("train sample: ", sum(nsents, trainsample), " sentences;  dev: ", sum(p -> nsents(p[2]), devdocs), " sentences")

  # model
  workdir = joinpath(HERE, "work", "train$(cfg["train_sents"])-" * join(tbs, "+") * "-" * cfg["events"] *
    (isempty(cfg["wclasses"]) ? "" : "-wc" * basename(rstrip(cfg["wclasses"], '/'))) *
    (isempty(cfg["suffixes"]) ? "" : "-sfx" * replace(cfg["suffixes"], ',' => "")) *
    (cfg["noise"] > 0 ? "-noise$(cfg["noise"])" : "") * (cfg["rootabsent"] == "true" ? "-rootabs" : "") *
    (cfg["relations"] == "true" ? "-rel" : "") * (cfg["headsdist"] == "true" ? "-sdist" : ""))
  mkpath(workdir)
  local model, lex
  modelname = cfg["model"]
  # how workers get the model: a serialized file, or built from the events (chain: its index is large)
  recipe = nothing
  if cfg["model"] in ("lattice", "chain")
    stampfile, lexfile, eventsdir = joinpath(workdir, "stamp"), joinpath(workdir, "lex.jls"), joinpath(workdir, "events")
    if isfile(stampfile) && read(stampfile, String) == srcstamp() && isfile(lexfile)
      lex = deserialize(lexfile)
      println("using cached events in $eventsdir")
    else
      lex = Lexicon()
      t = @elapsed record_events(trainsample, specs, lex, eventsdir; mode = Symbol(cfg["events"]))
      serialize(lexfile, lex)
      write(stampfile, srcstamp())
      println("recorded events in $(round(t; digits = 1)) s")
    end
    if cfg["model"] == "lattice"
      model = LatticeModel(eventsdir; maxorder = cfg["order"], minsupport = cfg["minsupport"],
        diffthreshold = cfg["diff"], wmothreshold = cfg["wmo"])
    else
      chainopts = (select = Symbol(cfg["select"]), tau = cfg["tau"], mincount = cfg["mincount"],
        leftovers = Symbol(cfg["leftovers"]), target = cfg["target"] == "true", maxrestarts = cfg["maxrestarts"],
        restart = Symbol(cfg["restart"]))
      cfg["screen"] > 0 && (chainopts = merge(chainopts, (screen = cfg["screen"], screenby = Symbol(cfg["screenby"]))))
      cfg["sample"] > 0 && (chainopts = merge(chainopts, (sample = cfg["sample"],)))
      cfg["climbleft"] != "deepest" && (chainopts = merge(chainopts, (climbleft = Symbol(cfg["climbleft"]),)))
      cfg["leftovers"] == "greedy" && (chainopts = merge(chainopts, (minctx = cfg["minctx"], greedyby = Symbol(cfg["greedyby"]))))
      cfg["leftovers"] == "greedy" && cfg["maxctx"] > 0 && (chainopts = merge(chainopts, (maxctx = cfg["maxctx"],)))
      cfg["leftovers"] == "greedy" && cfg["relevance"] > 0 && (chainopts = merge(chainopts, (relevance = cfg["relevance"],)))
      cfg["gate"] > 0 && (chainopts = merge(chainopts, (gate = cfg["gate"],)))
      cfg["pickfewest"] > 0 && (chainopts = merge(chainopts, (pickfewest = cfg["pickfewest"],)))
      cfg["contest"] == "true" && (chainopts = merge(chainopts, (contest = true,)))
      cfg["lockstep"] != "none" && (chainopts = merge(chainopts, (lockstep = Symbol(cfg["lockstep"]),)))
      cfg["stepscore"] != "abs" && (chainopts = merge(chainopts, (stepscore = Symbol(cfg["stepscore"]),)))
      cfg["redundant"] == "true" && (chainopts = merge(chainopts, (redundant = true,)))
      cfg["ctxsize"] != "min" && (chainopts = merge(chainopts, (ctxsize = Symbol(cfg["ctxsize"]),)))
      cfg["poolm"] > 0 && (chainopts = merge(chainopts, (poolm = cfg["poolm"], poollam = cfg["poollam"], poolgain = cfg["poolgain"], pairscore = Symbol(cfg["pairscore"]), pairfilter = cfg["pairfilter"] == "true")))
      model = nothing
      families = Tuple(Symbol.(split(cfg["climb"], ',')))
      recipe = (eventsdir, merge(chainopts, (jointmin = cfg["jointmin"], keepfrac = cfg["keepfrac"], maxcached = cfg["maxcached"],
        group = cfg["group"] == "true")),
        cfg["rings"], families)
      modelname = "chain(" * join(("$k=$v" for (k, v) in pairs(chainopts)), ";") *
        (isempty(cfg["rings"]) ? "" : ";climb=" * cfg["climb"]) * ")"
    end
    modelname *= "[events=$(cfg["events"])" * (isempty(cfg["wclasses"]) ? "" : ";wclasses=" * basename(rstrip(cfg["wclasses"], '/')) *
      (cfg["classleft"] == "true" ? "" : ";classleft=false")) *
      (isempty(cfg["suffixes"]) ? "" : ";suffixes=" * cfg["suffixes"]) * (cfg["noise"] > 0 ? ";noise=$(cfg["noise"])" : "") *
      (cfg["rootabsent"] == "true" ? ";rootabsent" : "") * (cfg["projective"] == "true" ? ";projective" : "") *
      (cfg["relations"] == "true" ? ";relations" : "") * (cfg["headsdist"] == "true" ? ";headsdist" : "") * "]"
  elseif cfg["model"] == "count"
    lex = Lexicon()
    model = train!(CountModel(), trainsample, specs, lex)
  else
    error("unknown model $(cfg["model"])")
  end
  build_s = time() - t0
  nlambda = model isa LatticeModel ? length(model.lambda) : model isa CountModel ? length(model.chosen) : 0
  (; tbs, idx, wc, sfx, devdocs, trainsample, workdir, model, lex, recipe, modelname, build_s, nlambda)
end

"""
Start the decoding workers, each holding WSPECS, WLEX and WMODEL.  They rebuild specs from a
copy of the index without its training-doc table (dev docs aren't in it, so leave-one-out
never applies to them).  Returns the temp dir to remove afterwards.
"""
function start_workers(cfg, s)
  (; idx, workdir, model, lex, recipe) = s
  rundir = mktempdir(workdir)
  serialize(joinpath(rundir, "model.jls"), recipe === nothing ? model : recipe)
  serialize(joinpath(rundir, "lex.jls"), lex)
  serialize(joinpath(rundir, "idx.jls"), CharIndex(idx.fwd, idx.rev, idx.gap, idx.saL, idx.saR, idx.cumL, idx.cumR,
    idx.prior, IdDict{Doc,Int}(), idx.maxlen, idx.mincount, idx.beta))
  addprocs(cfg["workers"]; exeflags = "--project=$HERE")
  Distributed.remotecall_eval(Main, workers(), :(using SeqLayers, Serialization))
  Distributed.remotecall_eval(Main, workers(), quote
    const WIDX = deserialize(joinpath($rundir, "idx.jls"))
    const WSPECS = default_specs(charindex = WIDX,
      wordclasses = $(isempty(cfg["wclasses"]) ? nothing : :(load_wordclasses($(cfg["wclasses"])))),
      suffixes = $(isempty(cfg["suffixes"]) ? Int[] : parse.(Int, split(cfg["suffixes"], ','))), noise = $(cfg["noise"]), rootabsent = $(cfg["rootabsent"] == "true"),
      projective = $(cfg["projective"] == "true"), relations = $(cfg["relations"] == "true"),
      headsdist = $(cfg["headsdist"] == "true"))
    const WLEX = deserialize(joinpath($rundir, "lex.jls"))
    const WMODEL = let m = deserialize(joinpath($rundir, "model.jls"))
      if m isa Tuple
        cm = ChainModel(m[1]; m[2]...)
        if length(m) >= 3 && !isempty(m[3])           # climbing rings
          R = deserialize(joinpath(m[3], "rings.jls"))
          T = [String(first(split(l, '\t'))) for l in eachline(joinpath(m[3], "targets.tsv"))]
          cm.climb = climb_table(T, R, WLEX; families = m[4])
        end
        if $(cfg["countcache"] == "true") && isempty(cm.climb)   # saved feature-set counts
          pc = SeqLayers.load_counts(m[1], $(s.recipe === nothing ? "" : eventsig(s.recipe[1])))
          pc !== nothing && (cm.pcounts = pc)
          cm.keepcounts = true
          cm.keepmin = $(cfg["countmin"])
        end
        $(cfg["classleft"] == "true") || (cm.noleft = SeqLayers.wclass_slots(WSPECS))
        cm
      else
        m
      end
    end
  end)
  rundir
end

function main(cfg)
  s = setup(cfg)
  (; tbs, devdocs, modelname, build_s, nlambda) = s

  # decode the dev docs in parallel
  rundir = start_workers(cfg, s)
  t1 = time()
  order = sortperm(devdocs; by = p -> length(p[2].text), rev = true)     # biggest first, for load balance
  counts = pmap(p -> (p[1], evaluate_counts([p[2]], WSPECS, WMODEL, WLEX)), devdocs[order])
  save_countcache(cfg, s)
  decode_s = time() - t1
  rmprocs(workers())
  rm(rundir; recursive = true)

  # report
  bytb = Dict(tb => Dict(k => 0 for k in keys(first(counts)[2])) for tb in [tbs; "ALL"])
  for (tb, c) in counts, (k, v) in c
    bytb[tb][k] += v
    bytb["ALL"][k] += v
  end
  runid = Dates.format(now(), "yyyymmdd-HHMMSS")
  newfile = !isfile(RESULTS)
  open(RESULTS, "a") do io
    newfile && println(io, "run,model,order,minsupport,diff,wmo,train_sents,dev_sents,treebank,words,token_f1,uas,las,sentence_f1,build_s,decode_s,nparams,note")
    println(rpad("treebank", 12), rpad("words", 8), "token_f1  uas     las     sent_f1")
    for tb in [tbs; "ALL"]
      s = eval_scores(bytb[tb])
      w = bytb[tb][:tok_gold]
      println(rpad(tb, 12), rpad(w, 8), join((rpad(round(x; digits = 4), 8) for x in s), ""))
      println(io, join((runid, repr(modelname), cfg["order"], cfg["minsupport"], cfg["diff"], cfg["wmo"],
          cfg["train_sents"], cfg["dev_sents"], tb, w, s.token_f1, s.uas, s.las, s.sentence_f1,
          round(build_s; digits = 1), round(decode_s; digits = 1), nlambda, repr(cfg["note"])), ","))
    end
  end
  println("build $(round(build_s; digits = 1)) s, decode $(round(decode_s; digits = 1)) s, run $runid → $RESULTS")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(parse_cfg(ARGS))
