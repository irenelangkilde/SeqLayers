#!/bin/zsh
# Greedy leftovers (minctx 2, relevance 0.25) on the slice with the installed code: first run fills
# the count cache with greedy's contexts, the second uses it.  Expect UAS 0.7385 / LAS 0.6961.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/greedy-rel025
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25)
run() { name=$1; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR\|count cache' $OUT/$name.log | grep -v '^--'; }
{
run greedy-rel025-fill
run greedy-rel025-warm
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
