#!/bin/bash
# The joint head property head_sdist (signed distance: direction and distance in one label).
# 1. nothing committed (headconf=1.0): its prediction accuracy vs head_dir, head_dist and the two
# both right; cold (new events).  2. headconf=0.99: committing confident head properties, head_sdist
# included; warm.  Base: gate 0.5 + relevance 0.25, rootabsent, noise=20.  Compare with
# gate05-rel-noise (0.7399 / 0.6992) and the earlier headconf 0.99 run (0.7406 / 0.6989, old base).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/sdist
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 headsdist=true)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|^pipeline\|Elapsed (wall\|recorded events\|using cached" $OUT/$name.log | grep -v "^--"; grep -A14 "^tag " $OUT/$name.log; }
{
run sdist-noncommit headconf=1.0
run sdist-conf099 headconf=0.99
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
