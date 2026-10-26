#!/bin/zsh
# Accum with restart (accumreset=true): each leftover's context starts from the chain + the leftovers taken so
# far at their most specific, generalized only as far as it needs; its generalizations are not carried forward.
# Same base as runs/accumwin (0.8465 / 0.7927 without restart).  Slice ceiling, Mac, 6 workers.  Waits for
# runs/accumwin.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/accumreset
until [ -f experiments/runs/accumwin/done ]; do sleep 60; done
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false \
  jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true \
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true \
  domains=product domeps=0 accumreset=true note=accumreset > $OUT/accumreset.log 2>&1
grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^score|ERROR" $OUT/accumreset.log > $OUT/summary.txt
touch $OUT/done
