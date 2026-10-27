#!/bin/zsh
# The head word's class as head-property decisions (headwc): head_wc30, and head_wc300 + head_wc30 -- generative
# factors (estimated from chosen pairs only) in the product.  Best ceiling setup (with head_trank): 0.8531 / 0.8020.
# Slice ceiling, Mac, 6 workers; each records its events.  Waits for runs/shrink.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/headwc
until [ -f experiments/runs/shrink/done ]; do sleep 60; done
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -E '^ALL|^score|ERROR|recorded events' $OUT/$name.log; grep -o 'tag|head_wc[0-9]*|[a-z]*' $OUT/$name.log | head -0; }
{
run hwc30 headwc=30
run hwc300-30 headwc=300,30
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
