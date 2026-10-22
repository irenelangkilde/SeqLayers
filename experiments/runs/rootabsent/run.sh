#!/bin/zsh
# Root candidate's candidate-side attachment items absent (root-ness carried once, by distance 0 and
# direction, instead of by every cand.* slot): best config (greedy, relevance 0.25), slice, cold.
# Compare with UAS 0.7385 / LAS 0.6961.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/rootabsent
/usr/bin/time -l julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 countmin=10000 rootabsent=true \
  note=rootabsent > $OUT/run.log 2>&1
touch $OUT/done
