#!/bin/zsh
# Greedy leftovers (minctx 2, relevance 0.25) on the slice, warm count cache, 3 workers instead of 6
# (6 overcommit the 24 GB machine).  Compare with greedy-rel025-mmap-warm: 4,881 s.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/greedy-rel025-w3
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=3 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 countmin=10000)
run() { name=$1; echo "=== $name  $(date)"; /usr/bin/time -l julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR\|count cache\|maximum resident\| real ' $OUT/$name.log | grep -v '^--'; }
{
run greedy-rel025-w3-warm
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
