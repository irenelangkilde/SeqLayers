#!/bin/zsh
# Example backoff traces for accumby=mintwc (base B): which proposal (ΔT / ΔW) took each step, and the steps past
# first support.  First non-punctuation error.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
julia --project=experiments experiments/ceiling_explain.jl model=chain select=maxdev events=gold dev_sents=5 workers=1 jointmin=10000 \
  headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
  poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 \
  headtrank=true accumby=mintwc nopunct=true docs=40 nsent=1 maxerr=2 maxlen=16 explaintop=2 detail=true \
  > experiments/runs/ceiling-explain/detail-mintwc.log 2>&1
