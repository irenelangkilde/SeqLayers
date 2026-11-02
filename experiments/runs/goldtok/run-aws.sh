#!/bin/bash
# Full decode on gold tokens and sentences (user 2026-11-02): the best decode (softheads + head_trank + posneg heads,
# 0.6936 / 0.6280, token F1 0.990, sentence F1 0.945) with the char layer's gap decisions from gold (goldtok=true):
# what segmentation errors cost.  14 workers; marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/goldtok
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
n=goldtok_decode_posnegheads
echo "=== $n  $(date)" > $OUT/summary-$TAG.txt
julia --project=experiments experiments/dev_eval.jl model=chain select=maxdev events=gold dev_sents=5 workers=14 \
  jointmin=10000 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
  poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true \
  stepscore=posneg posnegrestart=true posnegafter=true posnegonly=heads goldtok=true note=$n > $OUT/$n.log 2>&1
grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^build|ERROR|recorded events" $OUT/$n.log >> $OUT/summary-$TAG.txt
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
