# ci_table.jl  --  pairwise conditional independence of the features, from the recorded events (no
# decoding): for each attachment-style decision type, D[A, B] = how much feature B still changes o(x)
# once A is known, averaged over A's values (SeqLayers.ci_table).  Saved as ci_table.jls next to the
# events, where cidrop= / ciorder= runs read it.  Also the value-level table (A, a, B) -> D(a→B) for values a
# with at least vmin records in the sample (SeqLayers.ci_values), as ci_values.jls (ciorder=value).
#
#   julia --project=experiments experiments/ci_table.jl [dev_eval.jl options] [sample=200000] [vmin=200]

include(joinpath(@__DIR__, "dev_eval.jl"))
const S = SeqLayers
DEFAULTS["sample"] = 200_000
DEFAULTS["vmin"] = 200
DEFAULTS["domains"] = "off"; DEFAULTS["domeps"] = 1e-3           # (ceiling.jl options: accepted, unused here)
cfg = parse_cfg(["model=chain"; ARGS])
s = setup(cfg)
eventsdir, opts = s.recipe
specs = default_specs(wordclasses = s.wc, suffixes = s.sfx, noise = cfg["noise"], rootabsent = cfg["rootabsent"] == "true",
  projective = cfg["projective"] == "true", relations = cfg["relations"] == "true", headsdist = cfg["headsdist"] == "true")
m = ChainModel(eventsdir; opts...)
out = Dict{UInt64,Matrix{Float32}}()
outv = Dict{UInt64,Dict{Tuple{Int16,UInt64,Int16},Float32}}()
for k in eachindex(specs[2].decisions)
  tk = S.typekey(:token, k)
  t = @elapsed D = S.ci_table(m, tk; sample = cfg["sample"])
  D === nothing && continue
  out[tk] = D
  v = filter(!isnan, vec(D))
  q(p) = round(sort(v)[clamp(ceil(Int, p * length(v)), 1, length(v))]; digits = 3)
  println("$(specs[2].decisions[k].name): $(size(D, 1)) slots, $(round(t; digits = 1)) s; D quantiles 10% $(q(0.1)), 25% $(q(0.25)), 50% $(q(0.5)), 75% $(q(0.75)), 90% $(q(0.9))")
  for θ in (0.05, 0.1, 0.25)
    println("  pairs with D < $θ: $(count(<(θ), v)) of $(length(v))")
  end
  t = @elapsed V = S.ci_values(m, tk; sample = cfg["sample"], vmin = cfg["vmin"])
  outv[tk] = V
  nA = length(unique(k[1] for k in keys(V)))
  nv = length(unique(k[2] for k in keys(V)))
  println("  value table: $(length(V)) entries, $nv values (>= $(cfg["vmin"]) records) of $nA slots, $(round(t; digits = 1)) s")
end
path = joinpath(dirname(eventsdir), "ci_table.jls")
serialize(path, out)
println("→ $path")
pathv = joinpath(dirname(eventsdir), "ci_values.jls")
serialize(pathv, outv)
println("→ $pathv")
