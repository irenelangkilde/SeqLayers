#!/bin/zsh
# accum mindevsup + explainpool (0.8441 / 0.7913, mass ~26k) + depweight + prodprops=head_xpos,head_dir,head_dist.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/explainpool
julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 \
  headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
  poolmarginal=drop poolkeepgen=true leftsmooth=evidence projective=true domains=product domeps=0 headtrank=true dcstats=true \
  explainpool=true ctxmode=accum accumby=mindevsup depweight=true prodprops=head_xpos,head_dir,head_dist note=accum-mindevsup-pool-L \
  > $OUT/accum-mindevsup-pool-L.log 2>&1
grep -E '^ALL|^double|^explainpool|^score|ERROR' $OUT/accum-mindevsup-pool-L.log > $OUT/summary-L.txt
