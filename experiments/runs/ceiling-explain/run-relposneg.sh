#!/bin/bash
# Traces of 3 non-punctuation errors (any error, not only gold-fixable) on the relations ceiling + posneg heads setup
# (0.8731 / 0.8141), detailed chains and accum leftovers with units annotated.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs-rel
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
julia --project=experiments experiments/ceiling_explain.jl model=chain select=maxdev events=gold dev_sents=5 workers=1 jointmin=10000 headconf=1.0 \
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 redundant=true poolsigma=0.1 poolmarginal=drop \
  poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true relations=true maxcached=50000000 \
  domains=product domeps=0 stepscore=posneg posnegrestart=true posnegafter=true posnegonly=heads \
  nopunct=true goldfix=false docs=40 nsent=3 maxerr=2 maxlen=16 explaintop=2 detail=true > experiments/runs/ceiling-explain/detail-relposneg.log 2>&1
touch experiments/runs/ceiling-explain/done-relposneg
