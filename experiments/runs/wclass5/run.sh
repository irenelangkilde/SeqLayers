#!/bin/bash
# Word-class decisions (classes5: 30000/3000/300/30, in context and attachment features) on the
# slice, greedy leftovers (minctx 2, relevance 0.25), on AWS (r7g, 123 GB).  Compare with greedy
# without classes: UAS 0.7385 / LAS 0.6961.  Cold (records events), then warm.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/wclass5
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 countmin=10000
  wclasses=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5)
run() { name=$1; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|count cache\|Maximum resident\|Elapsed (wall" $OUT/$name.log | grep -v "^--"; }
{
run wclass5-cold
run wclass5-warm
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
