#!/bin/bash
# Relation runs with 2 workers and a smaller chain cache (memory): 1. relations + projective,
# 2. relations + pair steps (synergy).  Base: gate 0.5 + relevance 0.25, rootabsent, noise=20 (the
# relation events on this instance); warm.  Compare: relations 0.8048 / 0.7527.
# (src changed after the relation events were recorded, with byte-identical events: stamp refreshed.)
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/batch-rel
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs-rel
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=2 maxcached=50000000 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 relations=true)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|using cached\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{
run relations-projective projective=true
run relations-pairs pairsteps=true pairscore=synergy
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
