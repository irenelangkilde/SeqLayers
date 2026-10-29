#!/bin/zsh
# Remaining errors on the greedy mindev low-double-counting base (runs/mindev): ctxmode=greedy greedyby=mindev
# mindevstop=exhaust greedysel + depweight + prodprops=head_xpos,head_dir,head_dist, 0.8255 / 0.7713.
# Non-punctuation errors; chains, included leftovers, and every leftover's mindev path.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=1 jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  leftsmooth=evidence ctxmode=greedy projective=true domains=product domeps=0 headtrank=true
  greedyby=mindev mindevstop=exhaust greedysel=true depweight=true prodprops=head_xpos,head_dir,head_dist)
julia --project=experiments experiments/ceiling_explain.jl "${COMMON[@]}" nopunct=true \
  docs=40 nsent=4 maxerr=2 maxlen=16 explaintop=2 > experiments/runs/ceiling-explain/explain-mindev.log 2>&1
