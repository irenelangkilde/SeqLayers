#!/bin/zsh
# Tree rank both ways: the attachment item (treerank: 0.8524 / 0.7986 alone) and the head property (headtrank:
# 0.8531 / 0.8020 alone); base 0.8465 / 0.7927.  Slice ceiling, Mac, 6 workers; records new events.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/trank-both
julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false \
  jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true \
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true \
  domains=product domeps=0 treerank=true headtrank=true note=trank-both > $OUT/trank-both.log 2>&1
grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^score|ERROR|recorded events" $OUT/trank-both.log > $OUT/summary.txt
touch $OUT/done
