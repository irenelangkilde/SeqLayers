# rel_effects.jl  --  how much each context slot says about each decision type, from the recorded
# events alone (no decoding): the relevance gates' measure (split-half covariance of the change a
# slot's value makes to the odds; see slot_effects in chain.jl), for every decision type.
#
#   julia --project=experiments experiments/rel_effects.jl [dev_eval.jl options, e.g. relations=true noise=20 rootabsent=true]
#     top=25        slots listed per decision type (all relation slots are listed too)
#     minrecs=2000  labeled types: labels with fewer records are skipped (as GATEMIN)
#
# Attachment (one record per candidate): the slot's effect on o(this candidate is the head).
# Labeled decisions (one record per candidate label): the effect on o(label) per label, then the
# mean over labels weighted by their records, and the largest over labels (with the label).
# Relation slots (siblings, children, hypothetical siblings) are marked with *.

include(joinpath(@__DIR__, "dev_eval.jl"))
const S = SeqLayers
DEFAULTS["top"] = 25
DEFAULTS["minrecs"] = 2000
cfg = parse_cfg(["model=chain"; ARGS])
s = setup(cfg)
dir = s.recipe[1]
specs = default_specs(wordclasses = s.wc, suffixes = s.sfx, noise = cfg["noise"], rootabsent = cfg["rootabsent"] == "true",
  projective = cfg["projective"] == "true", relations = cfg["relations"] == "true", headsdist = cfg["headsdist"] == "true")
toks = specs[2]
ctx = toks.context
linkpaths = S.LINKPATHS[toks]
pathname(p) = join([string.(p.rels); string(p.feat)], '.')
const RELS = Set(string.(S.REL_KINDS))
isrel(name) = startswith(name, "hyp.") || any(r -> occursin(r, name), RELS)
const HYPNAMES = [string("hyp.", kind, ".", f) for kind in S.SIB_KINDS for f in (kind in S.DIR_KINDS ? (:upos, :deprel, :dist, :dir) : (:upos, :deprel, :dist))]
attachname(k) = k == 1 ? "distance" : k == 2 ? "direction" :
  (a = (k - 3) ÷ 2 + 1; a <= length(linkpaths) ? (isodd(k - 3) ? "cand " : "dep  ") * pathname(ctx[linkpaths[a]]) :
    get(HYPNAMES, k - 2 - 2 * length(linkpaths), "slot $k"))
labelname(dn, l) = dn == :deprel ? S.DEPRELS[l] : haskey(S.TAGSETS, dn) ? S.TAGSETS[dn][l] :
  haskey(S.ALLHEADSETS, dn) ? S.ALLHEADSETS[dn][l] : string(l)

t = @elapsed m = ChainModel(dir; mincount = 5, group = false)
println("$(s.modelname): $(length(m.recs)) records, loaded in $(round(t; digits = 1)) s\n")
for (k, d) in enumerate(toks.decisions)
  tk = S.typekey(:token, k)
  recs = get(m.typerecs, tk, Int32[])
  isempty(recs) && continue
  n = Int(m.recs[recs[1]].n)
  name(sl) = k == 1 ? attachname(sl) : sl <= length(ctx) ? pathname(ctx[sl]) : "slot $sl"
  t = @elapsed if !d.labeled
    p0 = S.priorodds(m, S.lpriorkey(tk, UInt32(0)))
    e, _ = S.slot_effects(m, recs, n, 0, p0)
    avg = [Float64(get(e, (Int32(0), Int16(sl)), 0f0)) for sl in 1:n]
    mx, mxl = copy(avg), fill("", n)
  else
    bylabel = Dict{UInt32,Vector{Int32}}()
    foreach(j -> push!(get!(() -> Int32[], bylabel, m.recs[j].label), j), recs)
    avg, wsum = zeros(n), 0.0
    mx, mxl = zeros(n), fill("", n)
    for (l, rl) in bylabel
      length(rl) >= cfg["minrecs"] || continue
      p0 = S.priorodds(m, S.lpriorkey(tk, l))
      e, _ = S.slot_effects(m, rl, n, 0, p0)
      w = length(rl)
      for sl in 1:n
        x = Float64(get(e, (Int32(0), Int16(sl)), 0f0))
        avg[sl] += w * x
        x > mx[sl] && (mx[sl] = x; mxl[sl] = labelname(d.name, l))
      end
      wsum += w
    end
    wsum > 0 && (avg ./= wsum)
  end
  order = sortperm(avg; rev = true)
  rank = invperm(order)
  shown = unique([order[1:min(cfg["top"], n)]; [sl for sl in order if isrel(name(sl))]])
  nrel = count(sl -> isrel(name(sl)), 1:n)
  println("== $(d.name): $(length(recs)) records, $n slots ($nrel relation slots), $(round(t; digits = 1)) s")
  println(rpad("", 2), rpad("slot", 30), lpad("rank", 6), lpad("mean effect", 13), lpad("max effect", 12), "  (label)")
  for sl in shown
    nm = name(sl)
    println(isrel(nm) ? "* " : "  ", rpad(nm, 30), lpad(rank[sl], 6), lpad(round(avg[sl]; digits = 4), 13),
      lpad(round(mx[sl]; digits = 4), 12), "  ", mxl[sl])
  end
  println()
end
