#!/bin/zsh
# Soft redundant pool: a leftover conditioned on pool features r blends its correction toward the one
# without them, weight W = prod exp(-d_r/sigma) (d_r: r's log-odds change at the chain's prefix, < tau).
# sigma 0.02 and 0.05.  Base: gate 0.5 + relevance 0.25 + redundant pool, rootabsent (no pair steps);
# slice, warm.  Waits for runs/pairsfirst.  Compare with runs/redundant (hard pool: 0.7413 / 0.6999).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/poolsigma
until [ -f experiments/runs/pairsfirst/done ]; do sleep 60; done
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR' $OUT/$name.log | grep -v '^--'; }
{
run poolsigma002 poolsigma=0.02
run poolsigma005 poolsigma=0.05
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
