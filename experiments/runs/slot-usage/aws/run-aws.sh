#!/bin/bash
# Slot usage without word classes, GUM smallest slice doc (the Mac runs the class version).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/slot-usage
/usr/bin/time -v julia --project=experiments experiments/slot_usage.jl model=chain select=maxdev events=gold dev_sents=5 \
  jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 tb=GUM docs=1 out=$OUT/noclasses.tsv > $OUT/noclasses.log 2>&1
touch $OUT/done-aws
