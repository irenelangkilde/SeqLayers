#!/bin/bash
# Updated detail trace (user 2026-11-11): the best non-forked ceiling, base B + kin + nn2 + relstrikes=2 (0.8865 / 0.8386),
# as detail-best-bioes.log was made (3 sentences, up to 2 non-punctuation errors each, 16 words at most), now with the
# head-property detail (each property whose top value isn't the gold head's: both values' chains and leftovers).
# Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/trace2
rm -f $OUT/done-$1
julia --project=experiments experiments/ceiling_explain.jl model=chain select=maxdev events=gold dev_sents=5 workers=1 jointmin=10000 headconf=1.0 \
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop \
  poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true \
  neighbor2=true kinitems=true headkin=left,right relstrikes=2 maxcached=50000000 \
  nopunct=true goldfix=false docs=40 nsent=3 maxerr=2 maxlen=16 explaintop=2 detail=true > $OUT/detail-kinnn2-rs2.log 2>&1
touch $OUT/done-$1
