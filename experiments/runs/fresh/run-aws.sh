#!/bin/bash
# Long-lived vs fresh decode workers (user 2026-11-11: the same full-EWT test docs ran ~3x slower late in a long run):
# the full-EWT model (best decode config minus oclass, relstrikes=1) on the first EWT test docs (dev_sents=40), one worker
# for all docs vs a fresh worker per doc (freshevery=1), side by side; profdecode prints each doc's time, its GC time and
# the worker's heap.  Needs First Worker's full-EWT events (re-stamped).  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/fresh
TAG=$1
rm -f "$OUT/done-$TAG"
W=experiments/work/train100000-EWT-gold-cdc3db987-wcclasses5-sfx3-rootabs-htrank-kin-hkinleftright-nn2
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp(\"experiments/corpora/ewt-v2.2.txt\"))"
C=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5
one() {
  n=fresh_$1
  julia --project=experiments experiments/dev_eval.jl model=chain select=maxdev events=gold jointmin=10000 leftovers=greedy \
    minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true \
    greedyby=odds leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true stepscore=posneg posnegrestart=true \
    posnegafter=true posnegonly=heads profdecode=true rescorestats=true candmemo=true attachwait=true rescoreevery=16 \
    neighbor2=true kinitems=true headkin=left,right suffixes=3 wclasses=$C wclevels=30,300 relstrikes=1 \
    corpus=experiments/corpora/ewt-v2.2.txt train_sents=100000 devsplit=test dev_sents=40 workers=1 $2 note=$n > $OUT/$n.log 2>&1
  { echo "=== $1 ($2)"; grep -E "^(EWT|ALL) |^build|ERROR|recorded events|using cached| chars: |profdecode: token.attach|cache emptied|t_chains|t_leftovers" $OUT/$n.log | sed 's/^ *From worker [0-9]*:\t*//'; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
one long freshevery=0 & sleep 60
one fresh freshevery=1 &
wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
