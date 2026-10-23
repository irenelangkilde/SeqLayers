#!/bin/bash
# Replaces run.sh after redundant-noise: pairs-total and pairs-synergy (pairs-min dropped: with every
# single competing, a pair scored by min(pair, members) can never win).  Same base as run.sh.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/pairsteps
while pgrep -f "note=[r]edundant-noise" > /dev/null; do sleep 30; done
# src changed after these events were recorded (chain.jl: pair scores; events unchanged): refresh the stamp
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 redundant=true)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|using cached" $OUT/$name.log | grep -v "^--"; }
{
echo "=== redundant-noise (run.sh)"; grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall" $OUT/redundant-noise.log | grep -v "^--"
run pairs-total pairsteps=true pairscore=total
run pairs-synergy pairsteps=true pairscore=synergy
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
