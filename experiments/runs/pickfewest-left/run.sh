#!/bin/bash
# Chain steps: of the 3 best-scoring features reaching tau, add the one with the fewest records
# (pickfewest=3), with gated, filtered greedy leftovers.  Base gate 0.5 + relevance 0.25,
# rootabsent, noise=20; slice, warm.  Compare with gate05-rel-noise (UAS 0.7399 / LAS 0.6992).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/pickfewest-left
/usr/bin/time -v julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 \
  pickfewest=3 note=pickfewest-left > $OUT/run.log 2>&1
touch $OUT/done
