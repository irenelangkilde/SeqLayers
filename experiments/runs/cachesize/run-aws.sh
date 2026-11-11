#!/bin/bash
# Why full-EWT scoring is ~12x slower (user 2026-11-11): the model-wide record cache (maxcached, record numbers; dev_eval
# default 200M) may overflow and be emptied over and over with 18.75M records.  The full-EWT model (best decode config
# minus oclass, relstrikes=1) on the first test docs (dev_sents=5), maxcached 200M vs 2000M side by side, 1 worker each;
# decision stats report the cache's emptyings and fill.  Needs First Worker's full-EWT events (re-stamped, not
# re-recorded: the source change since then is the CoNLL-U writer only).  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/cachesize
TAG=$1
rm -f "$OUT/done-$TAG"
W=experiments/work/train100000-EWT-gold-cdc3db987-wcclasses5-sfx3-rootabs-htrank-kin-hkinleftright-nn2
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp(\"experiments/corpora/ewt-v2.2.txt\"))"
C=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5
one() {
  n=cachesize_$1
  julia --project=experiments experiments/dev_eval.jl model=chain select=maxdev events=gold jointmin=10000 leftovers=greedy \
    minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true \
    greedyby=odds leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true stepscore=posneg posnegrestart=true \
    posnegafter=true posnegonly=heads profdecode=true rescorestats=true candmemo=true attachwait=true rescoreevery=16 \
    neighbor2=true kinitems=true headkin=left,right suffixes=3 wclasses=$C wclevels=30,300 relstrikes=1 \
    corpus=experiments/corpora/ewt-v2.2.txt train_sents=100000 devsplit=test dev_sents=5 workers=1 maxcached=$2 note=$n > $OUT/$n.log 2>&1
  { echo "=== $1 (maxcached $2)"; grep -E "^(EWT|ALL) |^build|ERROR|recorded events|using cached| chars: |profdecode: token.attach|cache emptied|t_chains|t_leftovers" $OUT/$n.log | sed 's/^ *From worker [0-9]*:\t*//'; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
one c200m 200000000 & sleep 60
one c2g 2000000000 &
wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
