#!/bin/bash
# Top-two lockstep chains vs independent maxdev chains, both without leftovers, on gate 0.5,
# rootabsent, noise=20; slice.  The first records events (cold), the second is warm from it.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/top2
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=none gate=0.5 countmin=10000 rootabsent=true noise=20)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|count cache\|Elapsed (wall\|using cached\|recorded events" $OUT/$name.log | grep -v "^--"; }
{
run noleft-indep
run noleft-top2 lockstep=top2
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
