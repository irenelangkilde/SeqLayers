#!/bin/bash
# Restarted chains (nchains; user's design 2026-11-05): after the chain, accum's first pick seeds a new chain over the
# pending features, up to nchains; chains 2.. are leftovers of chain 1, each counted in the context of the earlier ones
# (backed off as accum does); what remains pending is ordinary accum leftovers.  On base B + kin + nn2 (0.8782 / 0.8230):
# nchains 2 and 8 side by side, then 4 (at most two kin + nn2 ceilings building at once), 7 workers each.  dcstats
# reports how many chains each candidate built.  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/nchains
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true
  neighbor2=true kinitems=true headkin=left,right)
one() {
  n=nchains_$1
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=7 nchains=$1 note=$n > $OUT/$n.log 2>&1
  { echo "=== nchains $1"; grep -E '^ALL|^score|^chains built|recorded|ERROR' $OUT/$n.log; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
one 2 & sleep 30
one 8 &
wait
one 4
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
