#!/bin/zsh
# Remaining errors on the low-double-counting base (runs/dcladder): telescope + accumreset + depweight + product
# without head_upos / head_trank terms, 0.8330 / 0.7813.  Non-punctuation errors, detailed traces.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=1 jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true
  telescope=true accumreset=true depweight=true prodprops=head_xpos,head_dir,head_dist)
julia --project=experiments experiments/ceiling_explain.jl "${COMMON[@]}" nopunct=true \
  docs=40 nsent=1 sent="Watch watch it" maxerr=2 maxlen=16 explaintop=2 detail=true > experiments/runs/ceiling-explain/detail-lowdc-watch.log 2>&1
