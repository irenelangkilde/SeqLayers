#!/bin/bash
# As runs/gate05-rel (gate=0.5 plus relevance filter 0.25, rootabsent) with a 20-value random
# feature (noise=20); warm from gate05-noise.  Passes if accuracy matches gate05-rel.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/gate05-rel-noise
/usr/bin/time -v julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 \
  note=gate05-rel-noise > $OUT/run.log 2>&1
touch $OUT/done
