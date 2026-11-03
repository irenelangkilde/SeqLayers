#!/bin/bash
# Partial-gold decodes (user 2026-11-03), one after the other, after the decode profiles (runs/profdecode, waited for at
# most 4 h, then stopped): the best decode (softheads + head_trank + posneg heads; 0.6936 / 0.6280, gold tokens 0.7023 /
# 0.6326) with gold gaps + tags (level 2), then gold gaps + tags + head properties (level 3).  14 workers; marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/goldlevels
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
echo "=== queued $(date)" > $OUT/summary-$TAG.txt
for i in $(seq 1 240); do [ -f experiments/runs/profdecode/done-awsE ] && break; sleep 60; done
if [ ! -f experiments/runs/profdecode/done-awsE ]; then
  echo "=== profiles not done after 4 h: stopped $(date)" >> $OUT/summary-$TAG.txt
  echo "=== stopped after 4 h $(date)" >> experiments/runs/profdecode/summary-awsE.txt
  pkill -f "experiments/runs/profdecode/run-aws.sh"; pkill -f julia; sleep 20; pkill -9 -f julia; sleep 5
fi
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=14 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25
  gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true stepscore=posneg posnegrestart=true
  posnegafter=true posnegonly=heads)
for g in gap,tags gap,tags,heads; do
  n=goldlv_$(echo $g | tr ',' '+')
  echo "=== golddec=$g  $(date)" >> $OUT/summary-$TAG.txt
  julia --project=experiments experiments/dev_eval.jl "${COMMON[@]}" golddec=$g note=$n > $OUT/$n.log 2>&1
  grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^build|ERROR" $OUT/$n.log >> $OUT/summary-$TAG.txt
done
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
