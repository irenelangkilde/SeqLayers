#!/bin/bash
# Word classes in chains only (classleft=false: class slots never used as leftovers) on the slice,
# greedy leftovers (minctx 2, relevance 0.25), warm count cache from wclass5.  Compare with greedy
# without classes (UAS 0.7385 / LAS 0.6961) and with classes as leftovers too (0.7213 / 0.6726).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/wclass5-noleft
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 countmin=10000 classleft=false
  wclasses=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5)
run() { name=$1; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|count cache\|Maximum resident\|Elapsed (wall\|using cached\|recorded events" $OUT/$name.log | grep -v "^--"; }
{
run wclass5-noleft
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
