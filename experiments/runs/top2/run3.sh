#!/bin/bash
# Max-min2 chain steps (each comparison smoothed toward its own stage), no leftovers, independent
# chains and top-two freezing; same base (gate 0.5, rootabsent, noise=20).  Waits for run2.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/top2
until [ -f $OUT/done2 ]; do sleep 30; done
# src changed after these events were recorded (chain.jl only): refresh the stamp so they are reused
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=none gate=0.5 countmin=10000 rootabsent=true noise=20 stepscore=maxmin2)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|using cached\|recorded events" $OUT/$name.log | grep -v "^--"; }
{
run noleft-maxmin2
run noleft-maxmin2-top2 lockstep=top2
echo "=== all done $(date)"
} > $OUT/summary3.txt 2>&1
touch $OUT/done3
