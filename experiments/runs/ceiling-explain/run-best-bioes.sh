#!/bin/bash
# Traces of 3 non-punctuation errors (any error) on the best single-chain ceiling, base B + idxeq + kin + nn2 + bioes
# (0.8810 / 0.8251), detailed chains and accum leftovers, with the kinship / BIOES / neighbor2 items named (user
# 2026-11-03).  One process, maxcached 50M (it shares First Worker with the kin + nn2 fast decode).  Marks done-best-bioes.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/ceiling-explain
rm -f $OUT/done-best-bioes
julia --project=experiments experiments/ceiling_explain.jl model=chain select=maxdev events=gold dev_sents=5 workers=1 jointmin=10000 headconf=1.0 \
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop \
  poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true \
  idxeq=true neighbor2=true kinitems=true headkin=left,right bioes=true maxcached=50000000 \
  nopunct=true goldfix=false docs=40 nsent=3 maxerr=2 maxlen=16 explaintop=2 detail=true > $OUT/detail-best-bioes.log 2>&1
touch $OUT/done-best-bioes
