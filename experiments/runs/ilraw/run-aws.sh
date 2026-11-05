#!/bin/bash
# Interleave with raw leftover scores (user 2026-11-04): in the interleaved chain, a leftover competes with the chain step
# by log o(x | C+X) - log o(x | C) in its context with the chain step's own smoothing (ilscore=raw) instead of its
# evidence-shrunk correction (corr); taken, it still adds the correction.  On the best ceiling, base B + kin + nn2 +
# prodnorm=prior (0.8838 / 0.8296): interleave off / corr / raw, side by side, 5 workers each (cached kin + nn2 events).
# Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/ilraw
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true
  neighbor2=true kinitems=true headkin=left,right prodnorm=prior)
one() {
  n=ilraw_$2
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=5 $1 note=$n > $OUT/$n.log 2>&1
  { echo "=== $2"; grep -E '^ALL|^score|recorded|ERROR' $OUT/$n.log; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
one "" off & sleep 30
one "interleave=true" corr & sleep 30
one "interleave=true ilscore=raw" raw &
wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
