#!/bin/bash
# Pair steps: maxdev chains may add a precomputed pair (joint tables, jointmin 10000) in one step.
# Scored by total change, then by min(pair, each member).  Base: gate 0.5 + relevance 0.25 +
# redundant pool, rootabsent, noise=20; slice, warm.  Compare with runs/redundant (0.7413 / 0.6999,
# Mac, no noise: the gates keep it out of attachment).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/pairsteps
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 redundant=true)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|using cached" $OUT/$name.log | grep -v "^--"; }
{
run redundant-noise
run pairs-total pairsteps=true pairscore=total
run pairs-min pairsteps=true pairscore=min
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
