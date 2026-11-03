#!/bin/bash
# Decode speed (user 2026-11-03): attachwait (an attachment scored once its own UPOS + XPOS are in and its neighbours
# exist, or wait only on a gap that may be a sentence end) with rescoreevery n = 1, 4, 16, on the base decode with
# candmemo, EWT + CHILDES (candmemo alone: 358 / 467 s, 0.7696 / 0.6716).  Side by side, 2 workers each; done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/attachwait
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=2 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25
  gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true stepscore=posneg posnegrestart=true
  posnegafter=true posnegonly=heads profdecode=true rescorestats=true devtb=EWT,CHILDES candmemo=true attachwait=true)
one() {
  n=aw_n$1
  julia --project=experiments experiments/dev_eval.jl "${COMMON[@]}" rescoreevery=$1 note=$n > $OUT/$n.log 2>&1
  { echo "=== rescoreevery=$1"; grep -E "^(CHILDES|EWT|ALL) |ERROR| chars: |cmemo|profdecode: token.attach|profdecode: char.gap" $OUT/$n.log | sed 's/^ *From worker [0-9]*:\t*//'; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
one 1 & sleep 20; one 4 & sleep 20; one 16 &
wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
