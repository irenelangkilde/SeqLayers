# gate_stats.jl  --  calibrate chain relevance gates: each attachment slot's average effect, and how
# many slots each threshold admits.  One process, no decoding.
#
#   julia --project=experiments experiments/gate_stats.jl dir=<events dir> [noise=0] [suffixes=] [mincount=5]
#   julia --project=experiments experiments/gate_stats.jl [dev_eval.jl options, e.g. noise=20]
#
# Without dir=, the events are those dev_eval.jl would use for the options (recorded if needed).
# Prints, for the attachment decision: every slot's average effect (see Relevance gates in
# chain.jl), and its largest effect under any gate value (and which); then for thresholds
# 0.02 .. 0.5: slots admitted always, gate values that open others, and the mean number of slots
# admitted per record over a sample of records.

using SeqLayers
const S = SeqLayers

args = Dict(split(a, '='; limit = 2) for a in ARGS)
if haskey(args, "dir")
  dir = args["dir"]
else
  include(joinpath(@__DIR__, "dev_eval.jl"))
  dir = setup(parse_cfg(["model=chain"; ARGS])).recipe[1]
end
noise = parse(Int, get(args, "noise", "0"))
sfx = isempty(get(args, "suffixes", "")) ? Int[] : parse.(Int, split(args["suffixes"], ','))
mincount = parse(Int, get(args, "mincount", "5"))

specs = default_specs(suffixes = sfx, noise = noise)
ctx = specs[2].context
linkpaths = [S.LINK_PATHS; 22 .+ (1:length(sfx)+(noise > 0))]
pathname(p) = join([string.(p.rels); string(p.feat)], '.')
slotname(k) = k == 1 ? "distance" : k == 2 ? "direction" :
  (a = (k - 3) ÷ 2 + 1; (isodd(k - 3) ? "cand " : "dep  ") * pathname(ctx[linkpaths[a]]))

t = @elapsed m = ChainModel(dir; mincount, group = true, gate = 1e-12)
println("model with gates (all effects kept) in $(round(t; digits = 1)) s")
tk = S.typekey(:token, 1)
g = m.gates[tk]
idname = Dict{Int32,String}()      # gate value ids -> "slot=value#" (values are hashes; show the slot)
best = Dict{Int16,Tuple{Float32,Int32}}()
# the largest effect under a gate value, by the gate value's size
sizes = (0, 500, 2000, 10000, 50000)
bestby = Dict(n => Dict{Int16,Float32}() for n in sizes)
for ((v, s), e) in g.effectby
  (haskey(best, s) && best[s][1] >= e) || (best[s] = (e, v))
  for n in sizes
    g.valcount[v] >= n && (bestby[n][s] = max(get(bestby[n], s, 0f0), e))
  end
end
slotofid(v) = Int(m.slotof[v])
println("\n", rpad("slot", 26), lpad("effect", 9), lpad("max under a gate", 18), "   (gate slot)",
  join(lpad.(["max, gate ≥$n" for n in sizes[2:end]], 16)))
for s in sortperm(g.effect; rev = true)
  b = get(best, Int16(s), (0f0, Int32(0)))
  println(rpad(slotname(s), 26), lpad(round(g.effect[s]; digits = 4), 9), lpad(round(b[1]; digits = 4), 18),
    "   ", rpad(b[2] == 0 ? "" : slotname(slotofid(b[2])), 26),
    join(lpad.([string(round(get(bestby[n], Int16(s), 0f0); digits = 3)) for n in sizes[2:end]], 16)))
end
recs = m.typerecs[tk]
sample = recs[1:max(1, length(recs) ÷ 20_000):end]
println("\n", rpad("gate", 8), lpad("always", 8), lpad("gate values", 13), lpad("mean admitted/record", 22), "  of $(g.nslots)")
for θ in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
  always = g.effect .>= θ
  opens = Dict{Int32,Vector{Int}}()
  for ((v, s), e) in g.effectby
    (!always[s] && e >= θ && g.valcount[v] >= S.GATEMIN) && push!(get!(() -> Int[], opens, v), s)
  end
  tot = 0
  for j in sample
    r = m.recs[j]
    ok = copy(always)
    for s in g.gateslots
      v = m.ids[r.start+s-1]
      v == 0 && continue
      foreach(t -> ok[t] = true, get(opens, v, Int[]))
    end
    tot += count(ok)
  end
  println(rpad(θ, 8), lpad(count(always), 8), lpad(length(opens), 13), lpad(round(tot / length(sample); digits = 1), 22))
end
