#!/bin/bash
# Full decode (slice) with soft head properties (softheads) and the tree rank as a head property (headtrank:
# head_trank, a soft factor in each attachment candidate's score like the other head properties).  Soft heads + head_trank.
# Setup: soft pool + greedyby=odds + leftsmooth=evidence + accum + projective; greedy decoder, 12 workers; records
# the -htrank events first.  Compare: softheads alone 0.6825 / 0.6197; plain greedy 0.6766 / 0.5999.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/softheads-htrank
/usr/bin/time -v julia --project=experiments experiments/dev_eval.jl model=chain select=maxdev events=gold dev_sents=5 workers=12 \
  jointmin=10000 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
  poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true \
   note=softheads-htrank > $OUT/softheads-htrank.log 2>&1
grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^build|ERROR|recorded events|Elapsed \(wall|Maximum resident" $OUT/softheads-htrank.log > $OUT/summary.txt
touch $OUT/done
