#!/bin/bash
# Full v2.2 EWT measurement (user 2026-11-10, towards a CoNLL 2018-comparable run): train on all of v2.2 EWT train
# (12,543 sentences) with the best decode (kin + nn2 + suffixes=3 + word classes 30,300 + relstrikes=1, fast flags),
# decode the first 100 sentences of the v2.2 EWT test set with 2 workers.  Logs the machine's memory every minute
# (mem.log) -- for the events size, model build time, worker memory and time per token.  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/ewtfull
TAG=$1
rm -f "$OUT/done-$TAG"
( while true; do echo "$(date -u +%H:%M) $(free -g | sed -n 2p)" >> $OUT/mem-$TAG.log; sleep 60; done ) & MEMPID=$!
C=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5
n=ewtfull_measure
julia --project=experiments experiments/dev_eval.jl model=chain select=maxdev events=gold jointmin=10000 leftovers=greedy \
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true \
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true stepscore=posneg posnegrestart=true \
  posnegafter=true posnegonly=heads profdecode=true rescorestats=true candmemo=true attachwait=true rescoreevery=16 \
  neighbor2=true kinitems=true headkin=left,right suffixes=3 wclasses=$C wclevels=30,300 relstrikes=1 \
  corpus=experiments/corpora/ewt-v2.2.txt train_sents=100000 devsplit=test dev_sents=100 workers=2 note=$n > $OUT/$n.log 2>&1
kill $MEMPID
{ echo "=== $(date)"; grep -E "^(EWT|ALL) |^train sample|^build|ERROR|recorded events| chars: |profdecode: token.attach|grouped decisions|t_chains|t_leftovers" $OUT/$n.log | sed 's/^ *From worker [0-9]*:\t*//' | sort -u; du -sh experiments/work/*-c*/events 2>/dev/null; sort -k4 -n $OUT/mem-$TAG.log | tail -1; } > $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
