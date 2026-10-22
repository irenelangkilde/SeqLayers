#!/bin/zsh
# Adaptive leftover-context minimum: max(1, min(2, chain length + redundant pool)) -- short chains
# get leftovers conditioned on what they have.  With the redundant pool, then without (pool empty:
# max(1, min(2, chain length))).  Base: gate 0.5 + relevance 0.25, rootabsent; slice, warm.  Waits for
# runs/redundant.  Compare with gate05-rel (0.7399 / 0.6985) and runs/redundant (pool, fixed minimum 2).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/adaptive
until [ -f experiments/runs/redundant/done ]; do sleep 60; done
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"   # chain.jl only changed
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true ctxsize=adaptive)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -l julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR' $OUT/$name.log | grep -v '^--'; }
{
run adaptive-pool redundant=true
run adaptive-nopool
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
