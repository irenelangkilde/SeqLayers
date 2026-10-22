#!/bin/bash
# Acceptance test for relevance gates: as runs/gate05 (gate=0.5, relevance off, rootabsent) plus a
# 20-value random feature (noise=20, in context and attachment features).  Passes if accuracy and
# time barely differ from gate05.  Cold (records events).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/gate05-noise
/usr/bin/time -v julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0 gate=0.5 countmin=10000 rootabsent=true noise=20 \
  note=gate05-noise > $OUT/run.log 2>&1
touch $OUT/done
