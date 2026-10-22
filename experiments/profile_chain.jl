# profile_chain.jl  --  where does a chain model spend its time?  One process, one or a few dev docs.
#
#   julia --project=experiments experiments/profile_chain.jl [key=value ...]
#
# Takes dev_eval.jl's options (model=chain etc.; see its header) plus:
#   docs=1          the n smallest dev docs (from dev_sents's selection)
#   flat=300        print the flat profile, entries with at least this many samples (0 = no profile)
# and honours countcache (loading the events' saved counts first, saving new ones after).
#
# Prints the time, correct heads and labels (perfect-context ceiling, as ceiling.jl), how often the
# chain cache was emptied for space, peak memory; then scans and records read, split between main
# chains and greedy-leftover contexts (the diagnostic hooks in chain.jl); then pass-or-not decisions
# by trial mix; then the flat profile.

include(joinpath(@__DIR__, "dev_eval.jl"))
include(joinpath(@__DIR__, "ceiling_lib.jl"))
using Profile
const S = SeqLayers

DEFAULTS["docs"] = 1
DEFAULTS["flat"] = 300
DEFAULTS["headconf"] = 1.0
cfg = parse_cfg(ARGS)
cfg["model"] == "chain" || error("profile_chain.jl profiles the chain model (model=chain)")
s = setup(cfg)
idx = s.idx
widx = CharIndex(idx.fwd, idx.rev, idx.gap, idx.saL, idx.saR, idx.cumL, idx.cumR, idx.prior, IdDict{Doc,Int}(), idx.maxlen, idx.mincount, idx.beta)
specs = default_specs(charindex = widx, wordclasses = s.wc, suffixes = s.sfx, noise = cfg["noise"], rootabsent = cfg["rootabsent"] == "true")
eventsdir, opts = s.recipe
m = ChainModel(eventsdir; opts...)
cfg["classleft"] == "true" || (m.noleft = S.wclass_slots(specs))
if cfg["countcache"] == "true"
  m.keepcounts = true
  m.keepmin = cfg["countmin"]
  pc = S.load_counts(eventsdir, eventsig(eventsdir))
  pc !== nothing && (m.pcounts = pc)
  println("count cache: loaded $(length(m.pcounts)) feature sets")
end
docs = sort([d for (_, d) in s.devdocs]; by = d -> length(d.text))[1:cfg["docs"]]

const STATS = Dict(k => [0, 0] for k in (:chain, :greedy))        # [scans, records read]
const BASES = Dict(k => Int[] for k in (:chain, :greedy))
const MIX = Dict{Symbol,Vector{Tuple{Int,Int,Bool}}}(:chain => [], :greedy => [])
S.SCAN_HOOK[] = n -> (k = S.IN_GREEDY[] ? :greedy : :chain; STATS[k][1] += 1; STATS[k][2] += n; push!(BASES[k], n))
S.PASS_HOOK[] = (n, sh, skip) -> push!(MIX[S.IN_GREEDY[] ? :greedy : :chain], (n, sh, skip))

tot = Dict(:tok_n => 0, :uas => 0, :las => 0)
Profile.clear(); Profile.init(n = 10^7, delay = 0.01)
t = @elapsed for d in docs
  c = cfg["flat"] > 0 ? (@profile ceiling_counts(d, specs, m, s.lex; scoretags = false, headconf = cfg["headconf"])) :
    ceiling_counts(d, specs, m, s.lex; scoretags = false, headconf = cfg["headconf"])
  foreach(k -> (tot[k] += c[k]), keys(tot))
end
println("$(s.modelname): $(sum(d -> length(d.gold.head), docs)) tokens, $(round(t; digits = 1)) s; heads $(tot[:uas])/$(tot[:tok_n]), ",
  "labels $(tot[:las]); cache emptied $(S.CACHE_EMPTIED[]) times; max RSS $(round(Sys.maxrss() / 2^30; digits = 1)) GB\n")
for k in (:chain, :greedy)
  n, r = STATS[k]
  b = sort(BASES[k])
  println(rpad(string(k), 8), "scans $n, records read $r",
    isempty(b) ? "" : ", per scan median $(b[cld(end, 2)]), 90th pct $(b[ceil(Int, 0.9 * end)]), max $(b[end])")
end
println()
for k in (:chain, :greedy)
  v = MIX[k]
  isempty(v) && continue
  nskip = count(x -> x[3], v)
  long = [x[1] - x[2] for x in v if !x[3]]
  println(rpad(string(k), 8), "$(length(v)) pass decisions: skipped $nskip ($(round(100 * nskip / length(v); digits = 1))%); ",
    "passes forced by 1 long trial $(count(==(1), long)), 2: $(count(==(2), long)), 3+: $(count(>=(3), long)); median trials $(sort(first.(v))[cld(end, 2)])")
end
if cfg["countcache"] == "true"
  println("\ncount cache: $(length(m.newcounts)) new feature sets", merge_countcache(eventsdir, eventsig(eventsdir), m.newcounts))
end
if cfg["flat"] > 0
  println()
  Profile.print(format = :flat, sortedby = :count, mincount = cfg["flat"], noisefloor = 2)
end
