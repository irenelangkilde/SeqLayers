#!/bin/bash
# Timing the per-candidate score memo (candmemo=true): the rescoring diagnostic (runs/rescore: base EWT 509 s, CHILDES
# 584 s; kin + nn2 1,202 s / 1,164 s) again with the memo, side by side, 2 workers each.  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/candmemo
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=2 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25
  gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true stepscore=posneg posnegrestart=true
  posnegafter=true posnegonly=heads profdecode=true rescorestats=true devtb=EWT,CHILDES candmemo=true)
one() {
  n=cm_$2
  julia --project=experiments experiments/dev_eval.jl "${COMMON[@]}" $1 note=$n > $OUT/$n.log 2>&1
  { echo "=== $2"; grep -E "^(CHILDES|EWT|ALL) |ERROR|recorded events| chars: |^profdecode|^rescorestats" $OUT/$n.log | sed 's/^ *From worker [0-9]*:\t*//'; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
one "" base & sleep 30
one "neighbor2=true kinitems=true headkin=left,right" kinnn2 &
wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
