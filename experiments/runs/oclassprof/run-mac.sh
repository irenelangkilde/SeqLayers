#!/bin/bash
# Why oclass slows the decode ~4x (user 2026-11-09; suffixes=3 + oclass kin + nn2 decode: 10.5 h vs 2.7 h): base B fast
# decode on EWT only, 1 worker, suffixes=3 without / with oclass, profdecode + rescorestats (per decision type: time,
# scorings, and which changed rows set off rescores).  On the Mac.  CONFIGS overrides the configs (name:options).
# (--gcthreads=1: the first oclass run died in Julia's parallel GC, segfault, while the Mac was also running the tests.)
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
OUT=experiments/runs/oclassprof
COMMON=(model=chain select=maxdev events=gold dev_sents=5 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25
  gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true stepscore=posneg posnegrestart=true
  posnegafter=true posnegonly=heads profdecode=true rescorestats=true candmemo=true attachwait=true rescoreevery=16
  suffixes=3 devtb=EWT maxcached=50000000)
for c in ${CONFIGS:-"plain:" "oclass:oclass=true"}; do
  n=oclassprof_${c%%:*}
  julia --gcthreads=1 --project=experiments experiments/dev_eval.jl "${COMMON[@]}" workers=1 ${c#*:} note=$n > $OUT/$n.log 2>&1
done
touch $OUT/done
