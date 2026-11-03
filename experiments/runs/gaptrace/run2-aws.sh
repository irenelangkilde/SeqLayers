#!/bin/bash
# Ceiling vs decode with the pruning filter labelled (user 2026-11-03): gap_trace.jl on the kin + nn2 model (decode
# speed-ups on), EWT + CHILDES + ESLSpok.  Marks done-gap2.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/gaptrace
rm -f $OUT/done-gap2
julia --project=experiments experiments/gap_trace.jl select=maxdev events=gold dev_sents=5 workers=1 jointmin=10000 leftovers=greedy \
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true \
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true headtrank=true stepscore=posneg posnegrestart=true posnegafter=true \
  posnegonly=heads neighbor2=true kinitems=true headkin=left,right softheads=true candmemo=true attachwait=true rescoreevery=16 \
  maxcached=50000000 devtb=EWT,CHILDES,ESLSpok examples=20 > $OUT/gap2-kinnn2.log 2>&1
touch $OUT/done-gap2
