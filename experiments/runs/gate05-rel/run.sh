#!/bin/zsh
# Relevance gates (gate=0.5) together with the leftover relevance filter (0.25), rootabsent; slice,
# warm.  Compare with rootabsent without gates (UAS 0.7382 / LAS 0.6961, 5,649 s cold) and with
# gates alone (0.7320 / 0.6913, 7,682 s).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/gate05-rel
/usr/bin/time -l julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true \
  note=gate05-rel > $OUT/run.log 2>&1
touch $OUT/done
