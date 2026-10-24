#!/bin/bash
# PP-attachment preferences from DepCC as attachment items (pptable): for a dependent with a case child
# P, the candidate's word-class and word rate of taking a P-phrase.  1. base + pptable (cold: new events),
# 2. base + pair steps (synergy) + pptable.  Base: gate 0.5 + relevance 0.25 + redundant pool,
# rootabsent; slice.  Waits for runs/batch-cal.  Compare: base 0.7413 / 0.6999 (Mac), pairs-synergy
# 0.7703 / 0.7296 (with noise=20, kept out of attachment by the gates).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/pptable
until [ -f experiments/runs/batch-cal/done ]; do sleep 60; done
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true
  pptable=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/pptable/pptable.tsv)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|using cached\|recorded events\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{
run pptable-base
run pptable-pairs pairsteps=true pairscore=synergy
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
