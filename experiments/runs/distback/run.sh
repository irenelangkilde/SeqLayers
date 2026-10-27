#!/bin/zsh
# Distance backs off to direction in leftover contexts (distback): the attachment's distance gets direction as its
# one generalization rung in the accum ladders (distance -> direction -> dropped).  Best ceiling setup: soft pool +
# greedyby=odds + leftsmooth=evidence + accum + projective + product domains (eps 0) + head_trank: 0.8531 / 0.8020.
# Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/distback
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false \
  jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true \
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true \
  domains=product domeps=0 headtrank=true distback=true note=distback > $OUT/distback.log 2>&1
grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^score|ERROR|recorded events" $OUT/distback.log > $OUT/summary.txt
touch $OUT/done
