#!/bin/zsh
# Record lists for chain prefixes whose counts come from the count cache were rebuilt on every use
# (materialize! kept a built list only when the prefix was already in the chain cache).  A/B on one
# doc after one warm-up doc: reuse=false (old) vs reuse=true (keep those lists).  Soft pool model
# (sigma 0.1, drop + keepgen), gate 0.5 + relevance 0.25, redundant, rootabsent; Mac, one process.
# Waits for runs/pprel.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/profile-reuse
until [ -f experiments/runs/pprel/done ]; do sleep 60; done
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25 gate=0.5
  countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true docs=1 warm=1 flat=0)
for v in false true; do
  echo "=== reuse=$v  $(date)"
  julia --project=experiments experiments/profile_chain.jl "${COMMON[@]}" reuse=$v > $OUT/reuse-$v.txt 2>&1
  grep -E "warm-up|tokens, |record lists|scans" $OUT/reuse-$v.txt
done > $OUT/summary.txt 2>&1
touch $OUT/done
