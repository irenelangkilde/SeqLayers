#!/bin/bash
# Projectivity and the sibling/child relations on the current best base (gate 0.5 + relevance 0.25,
# rootabsent, noise=20); slice.  1. projective (warm; events unchanged by it), 2. relations (cold:
# new events), 3. both (warm from 2).  Compare with gate05-rel-noise (UAS 0.7399 / LAS 0.6992).
# (src changed after the base events were recorded, in ways that leave them unchanged with these
# options off -- engine/specs options, chain.jl; stamp refreshed.)
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/proj-rel
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|using cached\|recorded events\|Maximum resident" $OUT/$name.log | grep -v "^--" | head -12; }
{
run projective projective=true
run relations relations=true
run relations-projective relations=true projective=true
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
