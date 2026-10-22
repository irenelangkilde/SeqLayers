#!/bin/zsh
# Relevance gates as the only filter: gate=0.5 (from the noise calibration), leftover relevance
# filter off, root candidate items absent; slice, warm.  Compare with rootabsent without gates
# (UAS 0.7382 / LAS 0.6961) for accuracy and time (score 5,649 s cold there).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/gate05
/usr/bin/time -l julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0 gate=0.5 countmin=10000 rootabsent=true \
  note=gate05 > $OUT/run.log 2>&1
touch $OUT/done
