#!/bin/bash
# Open / closed class (oclass) and subtree BIOES attachment items (bioes) (user 2026-11-02), each and both, on base B
# (0.8531 / 0.8020) and on the best ceiling (idxeq + neighbor2 + kinitems + headkin, 0.8793 / 0.8248).  New specs:
# each records its own events.  Two batches of three, 4 workers each; marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/oclass-bioes
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=4 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
BEST="idxeq=true neighbor2=true kinitems=true headkin=left,right"
one() {
  n=ocb_$2_$(echo "$1" | tr ' ,' '_+' | tr '=' '-')   # (GNU tr reads '_-+' as a range)
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" $3 $1 note=$n > $OUT/$n.log 2>&1
  { echo "=== $2 $1"; grep -E '^ALL|^score|recorded|ERROR' $OUT/$n.log; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
one "oclass=true" B "" & sleep 20; one "bioes=true" B "" & sleep 20; one "oclass=true bioes=true" B "" & wait
one "oclass=true" best "$BEST" & sleep 20; one "bioes=true" best "$BEST" & sleep 20; one "oclass=true bioes=true" best "$BEST" & wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
