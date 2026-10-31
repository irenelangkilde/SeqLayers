#!/bin/zsh
# posneg heads (0.8579 / 0.8075) + leftmin2=2: leftovers need >= 2 chosen and >= 2 not-chosen records.  Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/posneg
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 \
  headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
  poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 \
  headtrank=true dcstats=true stepscore=posneg posnegrestart=true posnegafter=true posnegonly=heads leftmin2=2 \
  note=posnegheads-leftmin2-2 > $OUT/posnegheads-leftmin2-2.log 2>&1
grep -E '^ALL|^double|^posnegrestart|^score|ERROR' $OUT/posnegheads-leftmin2-2.log > $OUT/summary-leftmin2.txt
