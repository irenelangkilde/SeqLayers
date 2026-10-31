#!/bin/zsh
# posneg + posnegrestart (2026-10-30): restart the top candidate's chain seeded with its negative steps, until rank 1 is stable
# the chain's odds most (by tau); when none, the one lowering them most -- below the chain (posneg) or below the prior
# (posnegprior) -- then raising again.  signed = positive-only (stops when none raises by tau), as a control.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/posneg
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
{
for seed in own oldtop; do
  ss=posneg
  echo "=== posneg restart seed=$seed  $(date)"
  julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 \
    headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
    poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 \
    headtrank=true stepscore=$ss posnegrestart=true posnegseed=$seed dcstats=true note=stepscore-$ss-restart-$seed > $OUT/stepscore-$ss-restart-$seed.log 2>&1
  grep -E '^ALL|^posnegrestart|^score|ERROR' $OUT/stepscore-$ss-restart-$seed.log
done
} > $OUT/summary-restart.txt 2>&1
