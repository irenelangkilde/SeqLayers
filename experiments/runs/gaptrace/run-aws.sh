#!/bin/bash
# Ceiling vs decode (user 2026-11-03): gap_trace.jl on the kin + nn2 model (posneg heads; decode with softheads, candmemo,
# attachwait, rescoreevery 16), EWT + CHILDES.  On First Worker after the best-bioes ceiling trace; the kin + nn2 events
# dir is re-stamped first (the fast decode is reading it: re-recording there would break it).  Marks done-gap.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/gaptrace
rm -f $OUT/done-gap
until [ -f experiments/runs/ceiling-explain/done-best-bioes ]; do sleep 60; done
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank-kin-hkinleftright-nn2
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
julia --project=experiments experiments/gap_trace.jl select=maxdev events=gold dev_sents=5 workers=1 jointmin=10000 leftovers=greedy \
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true \
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true headtrank=true stepscore=posneg posnegrestart=true posnegafter=true \
  posnegonly=heads neighbor2=true kinitems=true headkin=left,right softheads=true candmemo=true attachwait=true rescoreevery=16 \
  maxcached=50000000 devtb=EWT,CHILDES examples=15 > $OUT/gap-kinnn2.log 2>&1
touch $OUT/done-gap
