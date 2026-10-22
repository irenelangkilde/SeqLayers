#!/bin/bash
# Hierarchical max-min chain steps (stepscore=maxmin: a feature scores the smallest |log y - log o|
# over the chain's odds so far; the largest such minimum is taken).  Base: gate 0.5 + relevance 0.25,
# rootabsent, noise=20; slice, warm.  Compare with gate05-rel-noise (UAS 0.7399 / LAS 0.6992).
# (src changed after these events were recorded: chain.jl and harness options only; stamp refreshed.)
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/maxmin
/usr/bin/time -v julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 \
  stepscore=maxmin note=maxmin > $OUT/run.log 2>&1
touch $OUT/done
