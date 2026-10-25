#!/bin/bash
# Relations + projective with product domains and no pruning (eps 0): on the soft pool model eps 0 was
# best (0.7779 / 0.7316); on relations + projective eps 1e-3 / 1e-4 lost a little (0.8686, 0.8665 vs
# 0.8706).  2 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/domains3
/usr/bin/time -v julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=2 \
  maxcached=50000000 tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 \
  rootabsent=true noise=20 relations=true projective=true domains=product domeps=0 note=relproj-product-0 > $OUT/relproj-product-0.log 2>&1
grep -A1 "^ALL\|^score\|ERROR\|^pipeline\|^domains\|Elapsed (wall" $OUT/relproj-product-0.log | grep -v "^--" > $OUT/summary.txt
touch $OUT/done
