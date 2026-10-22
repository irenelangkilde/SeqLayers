# slot_usage.jl  --  which slots do the chains and greedy leftovers use?  One process.
#
#   julia --project=experiments experiments/slot_usage.jl [key=value ...]
#
# Takes dev_eval.jl's options (model=chain, select=maxdev, leftovers=greedy, wclasses=... etc.) plus:
#   tb=GUM      only this treebank's slice docs ("" = all)
#   docs=1      the n smallest of those docs
#   out=        write the table as TSV here too
# Runs the perfect-context ceiling (as ceiling.jl, tags not scored) with the chain trace hooks on,
# and counts per decision type (attach, deprel) and slot:
#   chain      chain steps that added the slot (over all candidates scored)
#   first      chains whose first step was the slot
#   left_try   leftovers tried (past the relevance filter)
#   left_in    leftovers included (context of at least minctx)
#   ctx        times the slot was chosen as part of a leftover's context
#   |corr|     summed absolute correction of the slot's included leftovers
# Slots: attachment records are [distance, direction, then dependent and candidate value of each
# link path]; label decisions' are the token context paths in order.

include(joinpath(@__DIR__, "dev_eval.jl"))
include(joinpath(@__DIR__, "ceiling_lib.jl"))
const S = SeqLayers

DEFAULTS["docs"] = 1
DEFAULTS["tb"] = ""
DEFAULTS["out"] = ""
DEFAULTS["headconf"] = 1.0
cfg = parse_cfg(ARGS)
cfg["model"] == "chain" && cfg["select"] == "maxdev" || error("slot_usage.jl traces maxdev chains")
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
end
pool = [d for (tb, d) in s.devdocs if isempty(cfg["tb"]) || tb == cfg["tb"]]
docs = sort(pool; by = d -> length(d.text))[1:min(cfg["docs"], length(pool))]

# slot names
ctx = specs[2].context
pathname(p) = join([string.(p.rels); string(p.feat)], '.')
nlev = s.wc === nothing ? 0 : length(s.wc.sizes)
linkpaths = [S.LINK_PATHS; 22 .+ (1:nlev)]
attachname(k) = k == 1 ? "distance" : k == 2 ? "direction" :
  (a = (k - 3) ÷ 2 + 1; (isodd(k - 3) ? "cand " : "dep  ") * pathname(ctx[linkpaths[a]]))
slotname(key, k) = key[2] == 1 ? attachname(k) : pathname(ctx[k])
dname(key) = key[2] == 1 ? "attach" : key[2] == 2 ? "deprel" : string(specs[2].decisions[key[2]].name)

const COLS = (:chain, :first, :left_try, :left_in, :ctx, :abscorr)
U = Dict{Tuple{String,String},Dict{Symbol,Float64}}()
bump!(key, k, col, x = 1.0) = (d = get!(() -> Dict(c => 0.0 for c in COLS), U, (dname(key), slotname(key, k))); d[col] += x)
nsteps = Ref(0)
S.CHAIN_TRACE[] = (ev, data) -> begin
  ev == :step || return
  key = S.TRACE_KEY[]
  data.chosen == 0 && return
  bump!(key, data.chosen, :chain)
  # a first step starts from the prior: no earlier step for this candidate since the last :final
  nsteps[] == 0 && bump!(key, data.chosen, :first)
  nsteps[] += 1
end
let f = S.CHAIN_TRACE[]
  S.CHAIN_TRACE[] = (ev, data) -> (ev == :final ? (nsteps[] = 0) : f(ev, data))
end
S.LEFTOVER_TRACE[] = (slot, included, ctxslots, corr) -> begin
  key = S.TRACE_KEY[]
  bump!(key, slot, :left_try)
  included && (bump!(key, slot, :left_in); bump!(key, slot, :abscorr, abs(corr)))
  foreach(k -> k > 0 && bump!(key, k, :ctx), ctxslots)
end

tot = Dict(:tok_n => 0, :uas => 0, :las => 0)
t = @elapsed for d in docs
  c = ceiling_counts(d, specs, m, s.lex; scoretags = false, headconf = cfg["headconf"])
  foreach(k -> (tot[k] += c[k]), keys(tot))
end
println("$(s.modelname), $(isempty(cfg["tb"]) ? "all" : cfg["tb"]) docs: $(sum(d -> length(d.gold.head), docs)) tokens, ",
  "$(round(t; digits = 1)) s; heads $(tot[:uas])/$(tot[:tok_n]), labels $(tot[:las])\n")

lines = String[]
for dn in ("attach", "deprel")
  rows = sort([(k[2], v) for (k, v) in U if k[1] == dn]; by = r -> -(r[2][:chain] + r[2][:left_in]))
  isempty(rows) && continue
  sums = Dict(c => sum(r -> r[2][c], rows) for c in COLS)
  println(dn)
  println(rpad("slot", 22), join(lpad.(["chain", "%", "first", "%", "left_try", "left_in", "%", "ctx", "%", "|corr|"], 9)))
  pct(r, c) = sums[c] > 0 ? string(round(100 * r[2][c] / sums[c]; digits = 1)) : "-"
  for r in rows
    v = r[2]
    println(rpad(r[1], 22), join(lpad.([string(Int(v[:chain])), pct(r, :chain), string(Int(v[:first])), pct(r, :first),
      string(Int(v[:left_try])), string(Int(v[:left_in])), pct(r, :left_in), string(Int(v[:ctx])), pct(r, :ctx),
      string(round(v[:abscorr]; digits = 1))], 9)))
    push!(lines, join([dn; r[1]; [string(v[c]) for c in COLS]], '\t'))
  end
  println()
end
if !isempty(cfg["out"])
  open(cfg["out"], "w") do io
    println(io, join(["decision", "slot", map(string, COLS)...], '\t'))
    foreach(l -> println(io, l), lines)
  end
end
if cfg["countcache"] == "true"
  println("count cache: $(length(m.newcounts)) new feature sets", merge_countcache(eventsdir, eventsig(eventsdir), m.newcounts))
end
