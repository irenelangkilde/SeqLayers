#!/bin/zsh
# Attachment ceiling by training frequency of the dependent's and gold head's forms (depfreq,
# headfreq), for the best config: greedy leftovers (minctx 2, relevance 0.25), no classes, slice.
# Expect UAS 0.7385 / LAS 0.6961.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/freq-breakdown
/usr/bin/time -l julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 countmin=10000 note=freq-breakdown > $OUT/run.log 2>&1
touch $OUT/done
