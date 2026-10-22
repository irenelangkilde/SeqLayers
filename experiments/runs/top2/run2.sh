#!/bin/bash
# Max-min chain steps without leftovers (stepscore=maxmin, leftovers=none), same base as the top2
# runs (gate 0.5, rootabsent, noise=20); warm.  Compare with noleft-indep.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/top2
/usr/bin/time -v julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=none gate=0.5 countmin=10000 rootabsent=true noise=20 stepscore=maxmin \
  note=noleft-maxmin > $OUT/noleft-maxmin.log 2>&1
touch $OUT/done2
