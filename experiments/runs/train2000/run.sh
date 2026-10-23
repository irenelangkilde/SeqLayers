#!/bin/bash
# More training data: train_sents=2000 per treebank (4x the sample, about 18% of the training
# sentences), the best configuration without relations (gate 0.5 + relevance 0.25 + redundant pool,
# rootabsent); slice; cold (new events).  2 workers (memory grows with the events).  Waits for
# runs/pairsteps.  Compare with runs/redundant (train_sents=500: 0.7413 / 0.6999).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/train2000
until [ -f experiments/runs/pairsteps/done ]; do sleep 60; done
/usr/bin/time -v julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=2 \
  train_sents=2000 tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 \
  rootabsent=true redundant=true note=train2000 > $OUT/run.log 2>&1
grep -A1 "^ALL\|^score\|ERROR\|recorded events\|Elapsed (wall\|Maximum resident" $OUT/run.log | grep -v "^--" > $OUT/summary.txt
touch $OUT/done
