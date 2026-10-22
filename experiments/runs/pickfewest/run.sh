#!/bin/zsh
# Chain steps (Mac: the run without leftovers): of the 3 best-scoring features reaching tau (after the gates), add the one with the
# fewest records (pickfewest=3).  Without leftovers, then with gated, filtered greedy leftovers.
# Base gate 0.5, rootabsent; slice.  Waits for runs/contest.  Compare with independent chains
# without leftovers (0.7358 / 0.6864, with noise=20, which the gates exclude) and with leftovers
# (gate05-rel: 0.7399 / 0.6985).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/pickfewest
until [ -f experiments/runs/contest/done ]; do sleep 60; done
# src changed after these events were recorded (chain.jl and harness options only): refresh the stamp
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  gate=0.5 countmin=10000 rootabsent=true pickfewest=3)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -l julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR' $OUT/$name.log | grep -v '^--'; }
{
run pickfewest-noleft leftovers=none
# (pickfewest-left runs on AWS: runs/pickfewest-left)
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
