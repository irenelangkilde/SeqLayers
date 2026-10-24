#!/bin/bash
# Sibling and child relations: 1. each slot's mean / max effect on every decision type, from the
# events (rel_effects.jl); 2. how often chains and leftovers use each slot over all slice docs
# (slot_usage.jl, one process).  Model as runs/batch-rel relations (0.8048 / 0.7527): gate 0.5 +
# relevance 0.25, rootabsent, noise=20, relations.
# (src changed after the relation events were recorded, with byte-identical events: stamp refreshed.)
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/rel-analysis
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs-rel
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 maxcached=50000000 jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 relations=true)
echo "=== effects $(date)"
/usr/bin/time -v julia --project=experiments experiments/rel_effects.jl "${COMMON[@]}" > $OUT/effects.txt 2>&1
echo "=== usage $(date)"
/usr/bin/time -v julia --project=experiments experiments/slot_usage.jl "${COMMON[@]}" docs=10000 out=$OUT/usage.tsv > $OUT/usage.txt 2>&1
echo "=== all done $(date)"
touch $OUT/done
