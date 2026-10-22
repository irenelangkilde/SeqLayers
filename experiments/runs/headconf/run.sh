#!/bin/bash
# Pipeline ceiling with predicted head properties: predict head_upos/xpos/dir/dist with the
# attachment hidden, commit those at least `headconf` sure, narrow strictly, infer or choose.
# Best config (greedy, relevance 0.25), slice; 0.99 then 0.9 (the second warm from the first).
# Compare with nothing committed (UAS 0.7385 / LAS 0.6961) and all gold (0.9838 / 0.900).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/headconf
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000
  leftovers=greedy minctx=2 relevance=0.25 countmin=10000)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|count cache\|^pipeline\|Elapsed (wall\|using cached\|recorded events" $OUT/$name.log | grep -v "^--"; }
{
run headconf099 headconf=0.99
run headconf090 headconf=0.9
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
