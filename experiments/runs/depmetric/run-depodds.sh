#!/bin/zsh
# accumby=depodds: the per-leftover walk moves by |corr(E|∅)| * (dependence lost) + |log o(context') - T| (log odds); alone and with depweight.
# 1/2 per cell).  accumby=dep: each leftover walks its own generalization path, each step the move giving up the least
# dependence ρ(E, a) - ρ(E, a's next rung); depweight: correction times 1 - max ρ(E, a) over counted features (chain,
# taken leftovers) outside E's context; greedy: greedyby=dep (ρ) / depcorr (ρ * |Δ correction|) vs odds.
# Base: best ceiling setup (head_trank) 0.8531 / 0.8020 (accum).  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/depmetric
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  leftsmooth=evidence projective=true domains=product domeps=0 headtrank=true)
{
for v in "ctxmode=accum greedyby=odds accumby=depodds" "ctxmode=accum greedyby=odds accumby=depodds depweight=true"; do
  n=$(echo $v | tr ' =' '_-')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary-depodds.txt 2>&1
touch $OUT/done-depodds
