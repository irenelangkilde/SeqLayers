#!/bin/zsh
# The untried combination: pairs (synergy, slot order) on base B (accum + projective + product + head_trank,
# 0.8531 / 0.8020).  Pairs on the pre-head_trank setup: 0.8537 / 0.8030 (vs 0.8465 / 0.7927 without).
# Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/pairs-htrank
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 \
  headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
  poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 \
  headtrank=true dcstats=true pairsteps=true pairscore=synergy pairorder=slot note=pairs-htrank > $OUT/pairs-htrank.log 2>&1
grep -E '^ALL|^double|^score|ERROR' $OUT/pairs-htrank.log > $OUT/summary.txt
