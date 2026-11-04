#!/bin/bash
# Shape double counting (user 2026-11-04): the attachment chain's shape items (distance, direction, kinship) and the
# product's head-property terms describe the same tree shape.  On base B + kin + nn2 (0.8782 / 0.8230): 1 prodprops
# without head_kin*, head_trank, head_dist; 2 shapeitems=false (no shape items in the attachment chain; own events);
# 3 prodnorm=prior / attach (property terms divided by the value's prior / the attachment's own probability of it).
# Batch 1 records events (baseline, noshape: different dirs); batch 2 reuses the baseline's.  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/shape
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true
  neighbor2=true kinitems=true headkin=left,right)
one() {
  n=shape_$2
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=$3 $1 note=$n > $OUT/$n.log 2>&1
  { echo "=== $2"; grep -E '^ALL|^score|recorded|ERROR' $OUT/$n.log; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
one "" base 7 & sleep 30
one shapeitems=false noshape 7 &
wait
one prodprops=head_upos,head_xpos,head_dir prodprops 5 & sleep 30
one prodnorm=prior prior 5 & sleep 30
one prodnorm=attach attach 5 &
wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
