#!/bin/bash
# D-ranked pruning for speed (user 2026-11-10): leftovers are ~77% of attachment scoring time, chains ~23%.  Base B
# fast decode (it has ci_table.jls) on EWT only, 1 worker, profdecode + decision stats (chain / leftover time): no
# pruning, lefttopn 4 / 8, steptopn 8.  On the Mac.  CONFIGS overrides the configs (name:options, comma-separated).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
OUT=experiments/runs/topn
COMMON=(model=chain select=maxdev events=gold dev_sents=5 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25
  gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true stepscore=posneg posnegrestart=true
  posnegafter=true posnegonly=heads profdecode=true rescorestats=true candmemo=true attachwait=true rescoreevery=16
  devtb=EWT maxcached=50000000)
for c in ${CONFIGS:-"none:" "left4:lefttopn=4" "left8:lefttopn=8" "step8:steptopn=8"}; do
  n=topn_${c%%:*}
  julia --gcthreads=1 --project=experiments experiments/dev_eval.jl "${COMMON[@]}" workers=1 $(echo ${c#*:} | tr "," " ") note=$n > $OUT/$n.log 2>&1
  { echo "=== ${c%%:*}"; grep -E "^EWT |doc EWT|ERROR|profdecode: token.attach|t_chains_ms|t_leftovers_ms" $OUT/$n.log | sed 's/^ *From worker [0-9]*:\t*//'; } >> $OUT/summary.txt
done
touch $OUT/done
