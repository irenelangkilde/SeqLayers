#!/bin/bash
# Relations ceiling (relations + projective + accum + product, noise-20 relation events: 0.8769 / 0.8179) with the new
# best chains: posneg + restart after leftovers for attachment and head properties, abs for deprel/tags (posnegonly=heads).
# 2 workers (relation workers need 20-50 GB).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/rel-posnegheads
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs-rel
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 \
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 redundant=true poolsigma=0.1 poolmarginal=drop \
  poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true relations=true workers=2 maxcached=50000000 \
  domains=product domeps=0 stepscore=posneg posnegrestart=true posnegafter=true posnegonly=heads note=rel-posnegheads > $OUT/rel-posnegheads.log 2>&1
grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^score|ERROR" $OUT/rel-posnegheads.log > $OUT/summary.txt
touch $OUT/done
