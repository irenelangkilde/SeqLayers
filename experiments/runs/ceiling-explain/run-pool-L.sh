#!/bin/zsh
# Non-punctuation errors that gold head properties fix, on accum mindevsup + explainpool + depweight + prodprops.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
julia --project=experiments experiments/ceiling_explain.jl model=chain select=maxdev events=gold dev_sents=5 workers=1 jointmin=10000 \
  headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
  poolmarginal=drop poolkeepgen=true leftsmooth=evidence projective=true domains=product domeps=0 headtrank=true \
  explainpool=true ctxmode=accum accumby=mindevsup depweight=true prodprops=head_xpos,head_dir,head_dist \
  nopunct=true docs=40 nsent=2 maxerr=2 maxlen=16 explaintop=2 detail=true > experiments/runs/ceiling-explain/detail-pool-L.log 2>&1
