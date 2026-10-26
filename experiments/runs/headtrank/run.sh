#!/bin/zsh
# Tree rank as a head property (headtrank): a sixth head-property decision, head_trank (1, 2, 3, 4+, root), predicted
# from the token's own context; in the ceiling's product domains (eps 0) it is one more factor per candidate.
# Current best setup: soft pool + greedyby=odds + leftsmooth=evidence + accum + projective + product: 0.8465 / 0.7927.
# Slice ceiling, Mac, 6 workers; records new events (-htrank).  Waits for runs/treerank.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/headtrank
until [ -f experiments/runs/treerank/done ]; do sleep 60; done
julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false \
  jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true \
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true \
  domains=product domeps=0 headtrank=true note=headtrank > $OUT/headtrank.log 2>&1
grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^score|ERROR|recorded events" $OUT/headtrank.log > $OUT/summary.txt
grep -A1 "head_trank" experiments/ceiling_breakdown.csv | tail -4 >> $OUT/summary.txt
touch $OUT/done
