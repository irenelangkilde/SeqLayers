#!/bin/bash
# Leftover context rules (greedy leftovers), soft pool model (sigma 0.1, drop + keepgen), gate 0.5 +
# relevance 0.25, redundant, rootabsent; slice ceiling, 6 workers.  Compare: 0.7478 / 0.7044 (Mac).
#  ctxfromchain: each context starts as the deepest chain prefix with which the leftover is supported,
#                then extends greedily (close to the old backoff1 plus extension)
#  ctxsupport:   a context step needs context + leftover with at least 50 / 200 records (not 5)
#  greedyby=odds: context features picked by how much they move the context's own odds
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/ctxrules
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{
run ctxfromchain ctxfromchain=true
run ctxsupport50 ctxsupport=50
run ctxsupport200 ctxsupport=200
run greedyby-odds greedyby=odds
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
