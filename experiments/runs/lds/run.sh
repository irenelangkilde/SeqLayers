#!/bin/bash
# Limited-discrepancy search in the full decode (search=k): each segment's k least confident
# token-layer commits are retried with their runner-up from a copy of the state just before them;
# paths compared by summed (ldsscore=sum) or mean (mean) commit log probabilities.  Full decodes
# (dev_eval.jl: tokenization, tags, heads, labels from raw text) on the slice; model: soft pool
# sigma 0.1 + drop weak + keep generalizations, gate 0.5 + relevance 0.25, rootabsent; AWS, 12 workers
# (records the 500-sentence rootabsent events first).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/lds
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=12 jointmin=10000
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/dev_eval.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^build\|^search\|ERROR\|recorded events\|using cached\|Elapsed (wall\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{
run greedy
run lds4-sum search=4 ldsscore=sum
run lds4-mean search=4 ldsscore=mean
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
