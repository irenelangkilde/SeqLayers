#!/bin/bash
# Full decode with the new best ceiling setup's chains (posneg + restart after leftovers for attachment and head
# properties, abs for deprel/tags: posnegonly=heads) on the best decode (softheads + head_trank, 0.6870 / 0.6221).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/decode-posnegheads
NW=${1:-6}
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
julia --project=experiments experiments/dev_eval.jl model=chain select=maxdev events=gold dev_sents=5 workers=$NW \
  jointmin=10000 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
  poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true \
  stepscore=posneg posnegrestart=true posnegafter=true posnegonly=heads note=decode-posnegheads > $OUT/decode-posnegheads.log 2>&1
grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^build|ERROR|recorded events" $OUT/decode-posnegheads.log > $OUT/summary.txt
touch $OUT/done
