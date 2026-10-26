# dfs_diagnose.jl  --  why does the depth-first search fail?  Decodes the `docs` smallest slice docs with
# DFS (one process) logging every failure (SeqLayers.diagnose): which filter excluded each possible head of
# the token whose attachment failed, the token's committed head properties, and how many choice points back
# each was chosen.
#
#   julia --project=experiments experiments/dfs_diagnose.jl [dev_eval.jl options] docs=3 dfseps=0.05 dfsbudget=200 examples=15

include(joinpath(@__DIR__, "dev_eval.jl"))
const S = SeqLayers
DEFAULTS["docs"] = 3
DEFAULTS["examples"] = 15
DEFAULTS["domains"] = "off"; DEFAULTS["domeps"] = 1e-3           # (ceiling.jl options: accepted, unused here)
cfg = parse_cfg(["model=chain"; ARGS])
s = setup(cfg)
specs = default_specs(charindex = s.idx, wordclasses = s.wc, suffixes = s.sfx, noise = cfg["noise"], rootabsent = cfg["rootabsent"] == "true",
  projective = cfg["projective"] == "true", relations = cfg["relations"] == "true", headsdist = cfg["headsdist"] == "true",
  relbounded = cfg["relbounded"] == "true", relnone = cfg["relnone"] == "true", treerank = cfg["treerank"] == "true", headtrank = cfg["headtrank"] == "true")
eventsdir, opts = s.recipe
m = ChainModel(eventsdir; opts...)
pc = S.load_counts(eventsdir, eventsig(eventsdir))
pc !== nothing && (m.pcounts = pc)
(cfg["poolsigma"] > 0 || cfg["ctxmode"] != "greedy") && (m.generalize = S.generalization_slots(specs))
docs = sort([d for (_, d) in s.devdocs]; by = d -> length(d.text))[1:min(cfg["docs"], end)]
cfg["fwdcheck"] == "true" && (S.FWDCHECK[] = true)
S.DFS_DIAG[] = true
empty!(S.DFS_LOG)
t = @elapsed c = evaluate_counts(docs, specs, m, s.lex; search = DFS(eps = cfg["dfseps"], budget = cfg["dfsbudget"], jump = cfg["dfsjump"] == "true", order = Symbol(cfg["dfsorder"])))
println("$(length(docs)) docs, $(c[:tok_gold]) words, $(round(t; digits = 1)) s; UAS $(round(eval_scores(c).uas; digits = 4)); ", S.DFS_STATS)
L = S.DFS_LOG
println("\n$(length(L)) failures")
# the filter that excluded most of the token's possible heads, per failure; and heads excluded by each filter in all
dom = Dict{Symbol,Int}(); tot = Dict{Symbol,Int}()
for f in L
  k = argmax(f.why)
  dom[k] = get(dom, k, 0) + 1
  foreach(((r, n),) -> (tot[r] = get(tot, r, 0) + n), f.why)
end
println("\nmain cause (the filter excluding most possible heads):")
foreach(((r, n),) -> println("  ", rpad(r, 14), n, "  (", round(100n / length(L); digits = 1), "%)"), sort(collect(dom); by = x -> -x[2]))
println("\npossible heads excluded, by filter (all failures):")
foreach(((r, n),) -> println("  ", rpad(r, 14), n), sort(collect(tot); by = x -> -x[2]))
# committed head properties of the failing token: chosen at a choice point (how far back) or not
println("\nthe failing token's committed head properties:")
props = Dict{Symbol,Vector{Any}}()
for f in L, (n, b) in f.back
  push!(get!(() -> Any[], props, n), b)
end
for (n, bs) in sort(collect(props); by = x -> -length(x[2]))
  cp = [b for b in bs if b !== nothing]
  println("  ", rpad(n, 10), "committed in $(length(bs)) failures; at a choice point in $(length(cp))",
    isempty(cp) ? "" : ", choice points back: median $(sort(cp)[cld(end, 2)]), max $(maximum(cp))")
end
println("  no head property committed: $(count(f -> isempty(f.props), L))")
println("\nexamples:")
for f in L[1:max(1, length(L) ÷ cfg["examples"]):end][1:min(end, cfg["examples"])]
  println("  '", f.token, "': ", f.nheads, " possible heads; excluded by ", f.why, "; head properties ", f.props, "; back ", f.back, "; stack depth ", f.depth)
end
