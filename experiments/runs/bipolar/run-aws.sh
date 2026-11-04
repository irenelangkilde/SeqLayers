#!/bin/bash
# bipolar chain steps (user's design 2026-11-03): first step |Δ|, then the previous step's direction first (largest move
# that way by tau, else the largest the other way by tau).  Base B ceiling (0.8531 / 0.8020), side by side with abs
# (base B as is) and posneg (no restarts), 5 workers each.  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/bipolar
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
one() {
  n=bipolar_$2
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=5 $1 note=$n > $OUT/$n.log 2>&1
  { echo "=== $2"; grep -E '^ALL|^score|recorded|ERROR' $OUT/$n.log; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
one stepscore=abs abs & sleep 30
one stepscore=posneg posneg & sleep 30
one stepscore=bipolar bipolar &
wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
