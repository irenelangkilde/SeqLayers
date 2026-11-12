#!/bin/bash
# genred (user 2026-11-12): leftovers redundant with a chain feature on the same word, whatever the counts, dropped
# before the accum rounds -- word: any feature of the word; coarser: coarser than a chain feature (upos=ADV given
# xpos=RB), or finer only if upos / xpos.  On kin + nn2 + relstrikes=2 (0.8865 / 0.8386).  Mac, one at a time.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/genred
rm -f $OUT/done-mac
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true
  neighbor2=true kinitems=true headkin=left,right relstrikes=2)
echo "=== start $(date)" > $OUT/summary-mac.txt
for g in word coarser; do
  n=genred_$g
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=6 genred=$g note=$n > $OUT/$n.log 2>&1
  { echo "=== $g"; grep -E '^ALL|^score|recorded|ERROR|genred' $OUT/$n.log; } >> $OUT/summary-mac.txt
done
echo "=== all done $(date)" >> $OUT/summary-mac.txt
touch $OUT/done-mac
