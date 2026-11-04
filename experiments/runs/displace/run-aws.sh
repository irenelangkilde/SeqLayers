#!/bin/bash
# Displacement sweep (user 2026-11-03; gap2: 47 of 56 pruned gold heads blocked by an earlier wrong commit): base fast
# decode (softheads + head_trank + posneg heads, candmemo + attachwait + rescoreevery 16; 0.6992 / 0.6287 in 42 min)
# with displace = 0, 1, 2, side by side, 3 workers each.  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/displace
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25
  gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true stepscore=posneg posnegrestart=true
  posnegafter=true posnegonly=heads profdecode=true candmemo=true attachwait=true rescoreevery=16)
one() {
  n=displace_$2
  julia --project=experiments experiments/dev_eval.jl "${COMMON[@]}" workers=3 $1 note=$n > $OUT/$n.log 2>&1
  { echo "=== $2"; grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^build|ERROR|recorded events| chars: |profdecode: token.attach" $OUT/$n.log | sed 's/^ *From worker [0-9]*:\t*//'; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
one displace=0 d0 & sleep 30
one displace=1 d1 & sleep 30
one displace=2 d2 &
wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
