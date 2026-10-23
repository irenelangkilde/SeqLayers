#!/bin/zsh
# Soft pool by generalization quality: a pool feature g that generalizes a chain form w (its upos,
# xpos, basepos) is trusted by exp(-d_g/sigma), d_g = the chain's log-odds change when g replaces w;
# leftovers conditioned on g blend toward their context without it (poolmode=whole), or with d_g taken at
# w's own chain step and leftover corrections scaled by the product of weights (poolmode=local).
# sigma 0.1 and 0.3 each.  Base: gate 0.5
# + relevance 0.25 + redundant pool, rootabsent (no pair steps); slice, warm.  Waits for runs/pairsfirst.
# Compare with runs/redundant (hard pool: 0.7413 / 0.6999).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/poolsigma
until [ -f experiments/runs/pairsfirst/done ]; do sleep 60; done
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR' $OUT/$name.log | grep -v '^--'; }
{
run gensigma01 poolsigma=0.1
run gensigma03 poolsigma=0.3
run localsigma01 poolsigma=0.1 poolmode=local
run localsigma03 poolsigma=0.3 poolmode=local
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
