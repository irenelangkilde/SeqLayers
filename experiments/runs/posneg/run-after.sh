#!/bin/zsh
# posneg restart decided after the leftovers (posnegafter, 2026-10-31): rank by the full score (chain + leftovers),
# restart the top candidate's chain seeded with its chain negatives (or + its negative leftovers), recompute its
# leftovers, until rank 1 is stable.  Compare: restart before leftovers (own) 0.8565 / 0.7958; base B 0.8531 / 0.8020.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/posneg
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
{
for ln in false true; do
  n=stepscore-posneg-restart-after-leftneg-$ln
  echo "=== posneg restart after leftovers, leftover negatives in the seed: $ln  $(date)"
  julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 \
    headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
    poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 \
    headtrank=true stepscore=posneg posnegrestart=true posnegafter=true posnegleftneg=$ln dcstats=true note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^posnegrestart|^score|ERROR' $OUT/$n.log
done
} > $OUT/summary-after.txt 2>&1
