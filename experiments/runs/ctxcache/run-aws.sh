#!/bin/bash
# Cross-call context cache (ctxcache; 2026-11-10: 59% of accum context builds repeat an earlier one): the base B fast
# decode (candmemo + attachwait + rescoreevery 16, softheads, head_trank, posneg heads), all dev docs, ctxcache off vs
# 200000, side by side, 3 workers each (same machine conditions); decision stats on (chain / leftover time, context
# counts).  The parse must be identical.  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/ctxcache
TAG=$1
rm -f "$OUT/done-$TAG"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25
  gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true stepscore=posneg posnegrestart=true
  posnegafter=true posnegonly=heads profdecode=true rescorestats=true candmemo=true attachwait=true rescoreevery=16)
one() {
  n=ctxcache_$1
  julia --project=experiments experiments/dev_eval.jl "${COMMON[@]}" workers=3 ctxcache=$2 note=$n > $OUT/$n.log 2>&1
  { echo "=== $1"; grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^build|ERROR|recorded events| chars: |profdecode: token.attach|ctx_|t_chains|t_leftovers" $OUT/$n.log | sed 's/^ *From worker [0-9]*:\t*//'; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
one off 0 & sleep 30
one on 200000 &
wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
