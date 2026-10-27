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
#   pptable      PP-attachment table from DepCC (experiments/build_pptable.py; e.g.
#                juliaNLP/clusters/pptable/pptable.tsv): two attachment items for the dependent's
#                preposition and the candidate's class and word ("" = none); events apart
#   search       limited-discrepancy search in the full decode (src/search.jl): retry this many of each
#                segment's least confident commits with their runner-up (0 = plain easy-first)
#   ctxfromchain greedy leftovers: each context starts as the chain (as far as context + leftover stays
#                supported), then extends greedily: true | false
#   ctxsupport   greedy leftovers: a context step needs context + leftover with at least this many records
#                for the label (0 = only mincount)
#   leftsmooth   greedy leftovers: o(x|S∪E) as the odds in S∪E smoothed toward o(x|S) (odds) | Bayes'
#                rule with E's rates among the chosen / not-chosen records of S, each smoothed toward
#                E's rate in S (evidence)
#   ctxmode      greedy leftovers' contexts: greedy (as now) | backoff (start from the chain; while the
#                leftover is unsupported, generalize the chain feature -- form -> xpos/upos/classes ->
#                dropped -- that changes the chain's odds least) | limited (the same without dropping:
#                unsupported leftovers are rejected) | cover (greedy, from chain + pool + the chain
#                forms' generalizations; rejected unless every chain feature is in it itself or as
#                a generalization)
#   ctxattach    with ctxmode: use it for the attachment decision only (labels keep greedy contexts)
#   ctxscope     with ctxmode: all (every decision type) | heads (the attachment and the head-property
#                decisions; deprel and the word's own tags keep greedy contexts)
#   relbounded   with relations: left / right siblings don't cross the head; events apart
#   relnone      with relations: a relation that certainly doesn't exist is "none", apart from "not known
#                yet" (absent); events apart
#   accumwin     ctxmode=accum: each leftover's context is the one, among the first supported and this many
#                further generalization steps, where its |correction| is smallest (0: the first supported)
#   accumreset   ctxmode=accum: each leftover generalizes from the chain + earlier leftovers at their most
#                specific; its generalizations aren't carried forward: true | false
#   greedymin    greedy leftovers: each leftover's context is the prefix of its greedy growth (at least
#                minctx features) with the smallest |correction|: true | false
#   greedysel    greedy leftovers: taken in rounds, the largest |correction| first, each joining the
#                context pool (else in slot order): true | false
#   shrink       > 0: every chain and leftover estimate shrunk toward its parent's with a Beta prior of this many
#                pseudo-records (instead of the one-count smoothing)
#   distback     ctxmode=backoff/accum: in leftover contexts the attachment's distance generalizes to its
#                direction (distance -> direction -> dropped): true | false
#   driftsigma   ctxmode=accum: each correction weighted exp(-d/sigma), d = |log o(x | its context) - the running
#                estimate| (0 = off)
#   drifttau     ctxmode=accum: a leftover whose context has d > tau is left out (0 = off)
#   driftstep    ctxmode=accum: each correction weighted driftstep^(generalization and drop steps of its context)
#   treerank     attachment item per candidate: t's tree rank under it (1 + the candidate's dependents between
#                it and t: 1, 2, 3, 4+; root); events apart
#   headtrank    a sixth head-property decision, head_trank: the token's tree rank under its head (1, 2, 3, 4+,
#                root); narrows the attachment like the others; events apart
#   headwc       head-property decisions for the head word's class: head_wc300 / head_wc30 (e.g. 300,30), from
#                headwcdir (clusters/classes5); events apart
#   pairorder    with pairsteps: a chosen pair's members enter the chain in slot order (slot) or the one
#                with the larger single effect first (maxdev)
#   cidrop       ctxmode=accum: a context feature B may be dropped only if some feature A kept has
#                D(A→B) < cidrop (the pairwise table ci_table.jls next to the events: experiments/ci_table.jl)
#   wclevels     with wclasses: use only these class levels (sizes, e.g. 30,300); the others are kept out
#                of chains, leftovers and generalization ladders ("" = all)
#   classchain   with wclasses: may word-class slots be chosen in chains (false: stand-ins for forms
#                in leftover contexts only, never chains or leftovers): true | false
#   dfs          depth-first search with failures in the full decode (src/search.jl DFS): true | false
#   dfseps       DFS: a commit is a choice point if some other value has at least this probability
#   dfsbudget    DFS: failures allowed per segment before finishing greedily
#   fwdcheck     full decode: forward checking -- a head-property value is a candidate only if some possible
#                head has it (once no other head can appear): true | false
#   softheads    full decode: head properties never committed before their attachment; they weight its
#                candidates (product) and are derived from it: true | false
#   dfsepshead   DFS: the choice-point threshold for head-property decisions (default: dfseps)
#   dfsjump      DFS: conflict-directed backjumping (to the latest choice point among a failure's causes)
#   dfsorder     DFS: choice points' values in probability order (prob) or by the product score (product)
#   ldsscore     how search compares paths: sum | mean (of the commits' log probabilities)
#   ldsall       search may branch on char and sentence decisions too (default: token decisions only)
#   pprel        with pptable: each PP item compares the candidate's rate with the best of the token's
#                other candidates (log2 ratio buckets) instead of bucketing it alone; events apart
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
#   poolsigma    with redundant=true: a pool feature g that generalizes a chain form w (its upos, xpos,
#                class, ...) is trusted by exp(-d_g/sigma), d_g the change in the chain's log odds when g
#                replaces w; leftovers conditioned on g blend toward their context without it (0 = off)
#   poolmode     soft pool weights: whole (rebuild the chain with g for w; leftovers blend toward their
#                context without g) | local (at w's own chain step; leftover corrections scaled by the
#                product of their generalizations' weights)
#   poolmarginal with redundant=true: pool only features informative on their own (marginal |change in log
#                odds| >= tau) that the chain makes redundant: false (pool everything < tau given the chain) | true (the
#                weak ones stay leftovers, through the relevance filter) | drop (the weak ones are discarded)
#   poolkeepgen  with poolmarginal and poolsigma: generalizations of chain forms stay in the pool: true | false
#   poolscale    soft pool distance (whole mode): logodds (d_g) | z (d_g in standard errors of the chain's
#                final log odds, from its final prefix's counts; poolsigma then in SE units)
#   ctxsize      greedy leftovers' context size: min (at least minctx) | chain (exactly the chain's length) |
#                adaptive (at least max(1, min(minctx, chain length + redundant pool)))
#   stepscore    maxdev chain step score: abs (|log y - log x|) | signed (log y - log x: the feature
#                that raises the odds most) | maxmin (smallest |log y - log o| over the chain's odds so far) |
#                maxmin2 (the same with the counts smoothed toward each o in turn)
#   lockstep     maxdev chains: none (each candidate's chain runs to the end) | top2 (only the two
#                leading candidates advance; see run_top2 in chain.jl)
#   pairsteps    maxdev chain steps may add a precomputed pair (joint tables, jointmin) in one step,
#                scored by pairscore (total | min | increment | synergy: only pairs beating their better
#                member by tau, then on their full change); ties go to single features: true | false
#   pairsfirst   with pairsteps: features in any precomputed pair join chains only via a pair: true | false
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
#   tune_sents   held-out tuning sentences per treebank: the training sentences right after those
#                (ceiling.jl on=tune scores them, e.g. to calibrate the leftovers' weight)
#   leftalpha    weights of the greedy leftovers' corrections per decision type, from this TSV
#                (written by ceiling.jl calibrate=true); "" = all 1
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
  "wmo" => 1e-3, "train_sents" => 500, "tune_sents" => 100, "leftalpha" => "", "dev_sents" => 300, "workers" => 8, "note" => "",
  "select" => "clr_rms", "tau" => 0.1, "mincount" => 5, "jointmin" => 0, "sample" => 0.0, "rings" => "", "climb" => "cand", "climbleft" => "deepest", "minctx" => 1, "greedyby" => "correction", "maxctx" => 0, "relevance" => 0.0,
  "poolm" => 0, "poollam" => 0.0, "poolgain" => 0.1, "pairscore" => "total", "pairfilter" => "true", "keepfrac" => 1.0, "maxcached" => 200_000_000, "group" => "true", "countcache" => "true", "countmin" => 10_000, "wclasses" => "", "classleft" => "true", "suffixes" => "", "noise" => 0, "rootabsent" => "false", "projective" => "false", "relations" => "false", "headsdist" => "false", "pptable" => "", "pprel" => "false", "search" => 0, "dfs" => "false", "dfseps" => 0.05, "dfsbudget" => 200, "dfsjump" => "false", "fwdcheck" => "false", "softheads" => "false", "dfsepshead" => NaN, "dfsorder" => "prob", "ctxfromchain" => "false", "ctxsupport" => 0, "leftsmooth" => "odds", "ctxmode" => "greedy", "ctxattach" => "false", "ctxscope" => "all", "cidrop" => 0.0, "pairorder" => "slot", "accumwin" => 0, "accumreset" => "false", "greedymin" => "false", "greedysel" => "false", "driftsigma" => 0.0, "distback" => "false", "shrink" => 0.0, "headwc" => "", "headwcdir" => "/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5", "drifttau" => 0.0, "driftstep" => 0.0, "relbounded" => "false", "relnone" => "false", "treerank" => "false", "headtrank" => "false", "classchain" => "true", "wclevels" => "", "ldsscore" => "sum", "ldsall" => "false", "gate" => 0.0, "redundant" => "false", "ctxsize" => "min", "poolsigma" => 0.0, "poolmode" => "whole", "poolmarginal" => "false", "poolkeepgen" => "false", "poolscale" => "logodds", "stepscore" => "abs", "lockstep" => "none", "contest" => "false", "pickfewest" => 0, "pairsteps" => "false", "pairsfirst" => "false", "screen" => 0.0, "screenby" => "maxdev", "leftovers" => "backoff", "target" => "false",
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
  pp = isempty(cfg["pptable"]) ? nothing : load_pptable(cfg["pptable"], joinpath(dirname(dirname(cfg["pptable"])), "classes5"))
  sfx = isempty(cfg["suffixes"]) ? Int[] : parse.(Int, split(cfg["suffixes"], ','))
  hwl = isempty(cfg["headwc"]) ? Int[] : parse.(Int, split(cfg["headwc"], ','))
  hwc = isempty(hwl) ? nothing : something(wc, load_wordclasses(cfg["headwcdir"]))
  hwc === nothing || foreach(ds -> SeqLayers.add_head_wclass_gold!(ds, hwc, hwl), alltrain)
  specs = default_specs(charindex = idx, wordclasses = wc, suffixes = sfx, noise = cfg["noise"], rootabsent = cfg["rootabsent"] == "true",
    projective = cfg["projective"] == "true", relations = cfg["relations"] == "true",
    headsdist = cfg["headsdist"] == "true", pptable = pp, pprel = cfg["pprel"] == "true",
    relbounded = cfg["relbounded"] == "true", relnone = cfg["relnone"] == "true", treerank = cfg["treerank"] == "true", headtrank = cfg["headtrank"] == "true",
    headwc = hwl, headwcclasses = hwc)
  # same Doc objects as in the index, so their suffix-array features are leave-one-out
  trainsample = reduce(vcat, [firstsents(ds, cfg["train_sents"]) for ds in alltrain])
  # held-out tuning docs: the training docs right after the sample's
  tunedocs = [(tb, d) for (tb, ds) in zip(tbs, alltrain)
              for d in firstsents(ds[length(firstsents(ds, cfg["train_sents"]))+1:end], cfg["tune_sents"])]
  devdocs = [(tb, d) for (tb, f) in zip(tbs, devfiles) for d in firstsents(read_conllu(f), cfg["dev_sents"])]
  hwc === nothing || SeqLayers.add_head_wclass_gold!(last.(devdocs), hwc, hwl)
  println("train sample: ", sum(nsents, trainsample), " sentences;  dev: ", sum(p -> nsents(p[2]), devdocs), " sentences")

  # model
  workdir = joinpath(HERE, "work", "train$(cfg["train_sents"])-" * join(tbs, "+") * "-" * cfg["events"] *
    (isempty(cfg["wclasses"]) ? "" : "-wc" * basename(rstrip(cfg["wclasses"], '/'))) *
    (isempty(cfg["suffixes"]) ? "" : "-sfx" * replace(cfg["suffixes"], ',' => "")) *
    (cfg["noise"] > 0 ? "-noise$(cfg["noise"])" : "") * (cfg["rootabsent"] == "true" ? "-rootabs" : "") *
    (cfg["relations"] == "true" ? "-rel" * (cfg["relbounded"] == "true" ? "b" : "") * (cfg["relnone"] == "true" ? "n" : "") : "") * (cfg["headsdist"] == "true" ? "-sdist" : "") *
    (isempty(cfg["pptable"]) ? "" : cfg["pprel"] == "true" ? "-pprel" : "-pp") * (cfg["treerank"] == "true" ? "-trank" : "") * (cfg["headtrank"] == "true" ? "-htrank" : "") * (isempty(cfg["headwc"]) ? "" : "-hwc" * replace(cfg["headwc"], ',' => "")))
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
      cfg["ctxfromchain"] == "true" && (chainopts = merge(chainopts, (ctxfromchain = true,)))
      cfg["ctxsupport"] > 0 && (chainopts = merge(chainopts, (ctxsupport = cfg["ctxsupport"],)))
      cfg["leftsmooth"] != "odds" && (chainopts = merge(chainopts, (leftsmooth = Symbol(cfg["leftsmooth"]),)))
      cfg["shrink"] > 0 && (chainopts = merge(chainopts, (shrink = cfg["shrink"],)))
      (cfg["greedymin"] == "true" || cfg["greedysel"] == "true") &&
        (chainopts = merge(chainopts, (greedymin = cfg["greedymin"] == "true", greedysel = cfg["greedysel"] == "true")))
      cfg["ctxmode"] != "greedy" && (chainopts = merge(chainopts, (ctxmode = Symbol(cfg["ctxmode"]), ctxattach = cfg["ctxattach"] == "true",
        cidrop = cfg["cidrop"], accumwin = cfg["accumwin"], accumreset = cfg["accumreset"] == "true",
        driftsigma = cfg["driftsigma"], drifttau = cfg["drifttau"], driftstep = cfg["driftstep"],
        distback = cfg["distback"] == "true")))
      cfg["leftovers"] == "greedy" && cfg["relevance"] > 0 && (chainopts = merge(chainopts, (relevance = cfg["relevance"],)))
      cfg["gate"] > 0 && (chainopts = merge(chainopts, (gate = cfg["gate"],)))
      cfg["pairsteps"] == "true" && (chainopts = merge(chainopts, (pairsteps = true, pairscore = Symbol(cfg["pairscore"]),
        pairsfirst = cfg["pairsfirst"] == "true", pairorder = Symbol(cfg["pairorder"]))))
      cfg["pickfewest"] > 0 && (chainopts = merge(chainopts, (pickfewest = cfg["pickfewest"],)))
      cfg["contest"] == "true" && (chainopts = merge(chainopts, (contest = true,)))
      cfg["lockstep"] != "none" && (chainopts = merge(chainopts, (lockstep = Symbol(cfg["lockstep"]),)))
      cfg["stepscore"] != "abs" && (chainopts = merge(chainopts, (stepscore = Symbol(cfg["stepscore"]),)))
      cfg["redundant"] == "true" && (chainopts = merge(chainopts, (redundant = true, poolsigma = cfg["poolsigma"],
        poolmode = Symbol(cfg["poolmode"]), poolmarginal = Dict("false" => :off, "true" => :leftover, "drop" => :drop)[cfg["poolmarginal"]],
        poolkeepgen = cfg["poolkeepgen"] == "true", poolscale = Symbol(cfg["poolscale"]))))
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
      (cfg["relations"] == "true" ? ";relations" * (cfg["relbounded"] == "true" ? ";relbounded" : "") * (cfg["relnone"] == "true" ? ";relnone" : "") : "") * (cfg["headsdist"] == "true" ? ";headsdist" : "") *
      (isempty(cfg["pptable"]) ? "" : cfg["pprel"] == "true" ? ";pptable;pprel" : ";pptable") * (cfg["treerank"] == "true" ? ";treerank" : "") * (cfg["headtrank"] == "true" ? ";headtrank" : "") * (isempty(cfg["headwc"]) ? "" : ";headwc=" * cfg["headwc"]) * "]"
  elseif cfg["model"] == "count"
    lex = Lexicon()
    model = train!(CountModel(), trainsample, specs, lex)
  else
    error("unknown model $(cfg["model"])")
  end
  build_s = time() - t0
  nlambda = model isa LatticeModel ? length(model.lambda) : model isa CountModel ? length(model.chosen) : 0
  (; tbs, idx, wc, sfx, devdocs, tunedocs, trainsample, workdir, model, lex, recipe, modelname, build_s, nlambda)
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
      headsdist = $(cfg["headsdist"] == "true"),
      pptable = $(isempty(cfg["pptable"]) ? nothing : :(load_pptable($(cfg["pptable"]), $(joinpath(dirname(dirname(cfg["pptable"])), "classes5"))))),
      pprel = $(cfg["pprel"] == "true"), relbounded = $(cfg["relbounded"] == "true"), relnone = $(cfg["relnone"] == "true"), treerank = $(cfg["treerank"] == "true"), headtrank = $(cfg["headtrank"] == "true"),
      headwc = $(isempty(cfg["headwc"]) ? Int[] : parse.(Int, split(cfg["headwc"], ','))),
      headwcclasses = $(isempty(cfg["headwc"]) ? nothing : :(load_wordclasses($(isempty(cfg["wclasses"]) ? cfg["headwcdir"] : cfg["wclasses"])))))
    const WLEX = deserialize(joinpath($rundir, "lex.jls"))
    $(cfg["fwdcheck"] == "true") && (SeqLayers.FWDCHECK[] = true)
    $(cfg["softheads"] == "true") && (SeqLayers.SOFTHEADS[] = true)
    const WSEARCH = $(cfg["dfs"] == "true" ? :(SeqLayers.DFS(eps = $(cfg["dfseps"]), budget = $(cfg["dfsbudget"]), jump = $(cfg["dfsjump"] == "true"), order = $(QuoteNode(Symbol(cfg["dfsorder"]))), epshead = $(cfg["dfsepshead"]))) : cfg["search"] > 0 ? :(SeqLayers.LDS($(cfg["search"]); score = $(QuoteNode(Symbol(cfg["ldsscore"]))),
      tokenonly = $(cfg["ldsall"] != "true"))) : nothing)
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
        $(cfg["poolsigma"] > 0 || cfg["ctxmode"] != "greedy") && (cm.generalize = SeqLayers.generalization_slots(WSPECS))
        $(cfg["classchain"] == "true") || (cm.nochain = SeqLayers.wclass_slots(WSPECS))
        if $(!isempty(cfg["wclevels"]))                   # only these class levels: the others out of chains, leftovers and ladders
          keep = Set($(isempty(cfg["wclevels"]) ? Int[] : parse.(Int, split(cfg["wclevels"], ','))))
          drop = SeqLayers.wclass_slots(WSPECS; levels = [l for (l, sz) in enumerate($(s.wc === nothing ? Int[] : s.wc.sizes)) if !(sz in keep)])
          for (tk, sl) in drop
            cm.nochain[tk] = union(get(cm.nochain, tk, Set{Int}()), sl)
            haskey(cm.generalize, tk) && (cm.generalize[tk] = Dict(f => [g for g in gs if !(g in sl)] for (f, gs) in cm.generalize[tk]))
          end
        end
        $(cfg["cidrop"] > 0) && (cm.ci = deserialize(joinpath($(s.recipe[1]), "..", "ci_table.jls")))
        $(cfg["ctxscope"] == "heads") && (cm.ctxtypes = Set(SeqLayers.typekey(:token, k) for (k, d) in enumerate(WSPECS[2].decisions)
          if k == 1 || haskey(SeqLayers.ALLHEADSETS, d.name)))
        $(isempty(cfg["leftalpha"])) || (cm.leftalpha = Dict(parse(UInt64, split(l, '\t')[1]; base = 16) => parse(Float64, split(l, '\t')[3])
          for l in eachline($(cfg["leftalpha"])) if !startswith(l, '#')))
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
  counts = pmap(p -> (p[1], evaluate_counts([p[2]], WSPECS, WMODEL, WLEX; search = WSEARCH)), devdocs[order])
  if cfg["dfs"] == "true"
    st = [remotecall_fetch(() -> copy(SeqLayers.DFS_STATS), w) for w in workers()]
    println("dfs: ", Dict(k => sum(d[k] for d in st) for k in keys(first(st))))
  elseif cfg["search"] > 0
    st = [remotecall_fetch(() -> copy(SeqLayers.LDS_STATS), w) for w in workers()]
    println("search: ", Dict(k => sum(d[k] for d in st) for k in keys(first(st))))
  end
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
