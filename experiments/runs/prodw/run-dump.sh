#!/bin/bash
# Learned product weights (user 2026-11-12), step 1: dump each attachment's surviving candidates' product terms on the
# held-out tune split (300 training sentences per treebank after the train sample) and on dev, with the control
# (kin + nn2 + relstrikes=2, 0.8865 / 0.8386; the dev run is also the Mac control).  Mac, one at a time.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/prodw
rm -f $OUT/done-dump
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true
  neighbor2=true kinitems=true headkin=left,right relstrikes=2 tune_sents=300)
echo "=== start $(date)" > $OUT/summary-dump.txt
for on in tune dev; do
  n=dump_$on
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=6 on=$on proddump=$OUT/terms-$on note=$n > $OUT/$n.log 2>&1
  { echo "=== $on"; grep -E '^ALL|^score|recorded|ERROR|train sample' $OUT/$n.log; } >> $OUT/summary-dump.txt
done
echo "=== all done $(date)" >> $OUT/summary-dump.txt
touch $OUT/done-dump
