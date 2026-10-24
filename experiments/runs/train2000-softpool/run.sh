#!/bin/bash
# The best non-relation model with 4x the training data: redundant pool + soft pool by generalization
# quality (poolmode whole, sigma 0.1), gate 0.5 + relevance 0.25, rootabsent; train_sents=2000; slice;
# 3 workers; warm events (train2000).  Compare: same model at 500 sentences 0.7492 / 0.7058 (Mac),
# base at 2000 0.7641 / 0.7168.
# (src changed after these events were recorded, in options that leave them unchanged: stamp refreshed.)
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/train2000-softpool
W=experiments/work/train2000-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
/usr/bin/time -v julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=3 \
  train_sents=2000 tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 \
  rootabsent=true redundant=true poolsigma=0.1 poolmode=whole note=train2000-softpool > $OUT/run.log 2>&1
grep -A1 "^ALL\|^score\|ERROR\|recorded events\|using cached\|Elapsed (wall\|Maximum resident" $OUT/run.log | grep -v "^--" > $OUT/summary.txt
touch $OUT/done
