# trace_target.jl  --  what does target mode (top-two restarts) do differently from independent
# maxdev chains?  One process, the n smallest slice docs, the same model with target off and on.
#
#   julia --project=experiments experiments/trace_target.jl [dev_eval.jl options] [docs=1]
#
# For each mode and decision type (gap, attach, deprel): decisions scored, chain steps per candidate
# (features added, from the CHAIN_TRACE hook; in target mode this counts the passes after restarts
# too), how many decisions restarted (target mode: their candidates' final chains steer toward the
# odds reached when the top two changed), and accuracy of the perfect-context ceiling.

include(joinpath(@__DIR__, "dev_eval.jl"))
include(joinpath(@__DIR__, "ceiling_lib.jl"))
const S = SeqLayers

DEFAULTS["docs"] = 1
DEFAULTS["headconf"] = 1.0
cfg = parse_cfg(ARGS)
s = setup(cfg)
idx = s.idx
widx = CharIndex(idx.fwd, idx.rev, idx.gap, idx.saL, idx.saR, idx.cumL, idx.cumR, idx.prior, IdDict{Doc,Int}(), idx.maxlen, idx.mincount, idx.beta)
specs = default_specs(charindex = widx, wordclasses = s.wc, suffixes = s.sfx, noise = cfg["noise"], rootabsent = cfg["rootabsent"] == "true")
eventsdir, opts = s.recipe
docs = sort([d for (_, d) in s.devdocs]; by = d -> length(d.text))[1:cfg["docs"]]
dname(key) = key[1] == :char ? "gap" : key[2] == 1 ? "attach" : key[2] == 2 ? "deprel" : string(key[2])

for target in (false, true)
  m = ChainModel(eventsdir; merge(opts, (target = target,))...)
  if cfg["countcache"] == "true"
    pc = S.load_counts(eventsdir, eventsig(eventsdir))
    pc !== nothing && (m.pcounts = pc)
  end
  steps = Dict{String,Int}(); cands = Dict{String,Int}(); decs = Dict{String,Int}(); restarted = Dict{String,Int}()
  # count steps per decision; a decision restarted if a candidate's chain shrank between steps
  cur = Ref{Any}(nothing); depth = Dict{Any,Int}(); shrank = Ref(false)
  S.CHAIN_TRACE[] = (ev, data) -> begin
    ev == :step || return
    key = S.TRACE_KEY[]
    if key !== cur[]
      cur[] !== nothing && shrank[] && (restarted[dname(cur[])] = get(restarted, dname(cur[]), 0) + 1)
      cur[] = key; empty!(depth); shrank[] = false
      decs[dname(key)] = get(decs, dname(key), 0) + 1
    end
    d = dname(key)
    c = data.cand
    data.chosen == 0 && return
    steps[d] = get(steps, d, 0) + 1
    # a restart starts every chain from the prior again: the odds before the step equal a prior seen
    k = (c, round(data.before; sigdigits = 12))
    haskey(depth, k) && (shrank[] = true)
    depth[k] = 1
  end
  lefts = Dict{String,Vector{Int}}()                 # decision -> [leftovers tried, included]
  S.LEFTOVER_TRACE[] = (slot, included, ctx, corr) -> (v = get!(() -> [0, 0], lefts, dname(S.TRACE_KEY[])); v[1] += 1; v[2] += included)
  t = @elapsed begin
    tot = Dict(:tok_n => 0, :uas => 0, :las => 0, :gap_n => 0, :gap_ok => 0)
    for d in docs
      c = ceiling_counts(d, specs, m, s.lex; scoretags = false, headconf = cfg["headconf"])
      foreach(k -> (tot[k] += c[k]), keys(tot))
    end
  end
  cur[] !== nothing && shrank[] && (restarted[dname(cur[])] = get(restarted, dname(cur[]), 0) + 1)
  println("\n== target = $target: $(round(t; digits = 1)) s; heads $(tot[:uas])/$(tot[:tok_n]), labels $(tot[:las]), gaps $(tot[:gap_ok])/$(tot[:gap_n])")
  println(rpad("decision", 10), lpad("decisions", 11), lpad("chain steps", 13), lpad("steps/decision", 16), lpad("restarted", 11), lpad("leftovers try", 14), lpad("included", 14))
  for d in sort(collect(keys(decs)))
    lt = get(lefts, d, [0, 0])
    println(rpad(d, 10), lpad(decs[d], 11), lpad(get(steps, d, 0), 13), lpad(round(get(steps, d, 0) / decs[d]; digits = 2), 16),
      lpad(get(restarted, d, 0), 11), lpad(lt[1], 14), lpad(lt[2], 14))
  end
  S.CHAIN_TRACE[] = nothing; S.LEFTOVER_TRACE[] = nothing
end
