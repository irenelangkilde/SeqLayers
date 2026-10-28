#!/bin/zsh
# Sentences the accumby=dep (Yule's Q) ceiling gets wrong and the accum-odds base gets right, no punctuation
# in the error (token, gold head, predicted head); detailed trace under both models.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=1 jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true)
julia --project=experiments experiments/ceiling_explain.jl "${COMMON[@]}" accumby=dep contrast=accumby:odds nopunct=true \
  docs=40 nsent=2 maxerr=2 maxlen=18 explaintop=2 detail=true > experiments/runs/ceiling-explain/detail-depwrong.log 2>&1
