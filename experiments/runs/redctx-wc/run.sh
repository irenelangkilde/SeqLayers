#!/bin/bash
# Clusters as ordinary features (wclasses=classes5, may be chain features and leftovers), redundant
# features in the leftovers conditioning pool, leftover contexts exactly as long as the chain; on
# gate 0.5 + relevance 0.25, rootabsent; slice, cold.  Compare with runs/redctx (the same without
# classes) and with gate05-rel (0.7399 / 0.6992 with noise).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/redctx-wc
/usr/bin/time -v julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true \
  redundant=true ctxsize=chain wclasses=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5 \
  note=redctx-wc > $OUT/run.log 2>&1
touch $OUT/done
