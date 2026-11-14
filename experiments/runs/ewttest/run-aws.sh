#!/bin/bash
# Full UD v2.2 EWT test decode (user 2026-11-13), the fair CoNLL 2018 comparison: the full-EWT model (12,543 training
# sentences) with the cached-events setup -- fast kin + nn2 decode, suffixes=3, classes5 levels 30/300, relstrikes=1
# (dev 0.7374 / 0.6854) -- CoNLL-U output with one root per sentence (reattached).  Args: tag, shard (k/2), and
# optionally dev_sents (default 0 = all; prep runs use 1 just to record the events).  3 workers (~22 GB each).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/ewttest
TAG=$1; SHARD=$2; DS=${3:-0}
rm -f "$OUT/done-$TAG"
W=experiments/work/train100000-EWT-gold-cdc3db987-wcclasses5-sfx3-rootabs-htrank-kin-hkinleftright-nn2
# (the events stay valid across today's source changes: none alters event recording with these options)
[ -d $W/events ] && julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp(\"experiments/corpora/ewt-v2.2.txt\"))"
C=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5
n=ewttest_${TAG}
julia --project=experiments experiments/dev_eval.jl model=chain select=maxdev events=gold jointmin=10000 leftovers=greedy \
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true \
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true stepscore=posneg posnegrestart=true \
  posnegafter=true posnegonly=heads profdecode=true candmemo=true attachwait=true rescoreevery=16 \
  neighbor2=true kinitems=true headkin=left,right suffixes=3 wclasses=$C wclevels=30,300 relstrikes=1 \
  corpus=experiments/corpora/ewt-v2.2.txt train_sents=100000 devsplit=test dev_sents=$DS shard=$SHARD workers=3 \
  conllu_out=$OUT/$TAG note=$n > $OUT/$n.log 2>&1
{ echo "=== $TAG ($SHARD, dev_sents $DS)"; grep -E "^(EWT|ALL) |^build|ERROR|recorded events|using cached|count cache|conllu:" $OUT/$n.log | sed 's/^ *From worker [0-9]*:\t*//'; } > $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
