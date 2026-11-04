#!/bin/bash
# accum backoff floor (user 2026-11-04): the backoff never drops a context feature once only minctx (2) are left.  Base B
# ceiling (0.8531 / 0.8020) without and with it, side by side, 7 workers each; dcstats=true reports how many context
# features each taken leftover had.  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/accumfloor
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
one() {
  n=accumfloor_$2
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=7 $1 note=$n > $OUT/$n.log 2>&1
  { echo "=== $2"; grep -E '^ALL|^score|^accum context size|recorded|ERROR' $OUT/$n.log; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
one accumfloor=false off & sleep 30
one accumfloor=true on &
wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
