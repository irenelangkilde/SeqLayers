#!/bin/zsh
# Which slots do chains and greedy leftovers use, with and without word classes?  GUM's smallest
# slice doc (GUM lost 4 UAS with classes), both in parallel.  Tables in *.log, TSVs alongside.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/slot-usage
C=(model=chain select=maxdev events=gold dev_sents=5 jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 tb=GUM docs=1)
julia --project=experiments experiments/slot_usage.jl "${C[@]}" out=$OUT/noclasses.tsv > $OUT/noclasses.log 2>&1 &
julia --project=experiments experiments/slot_usage.jl "${C[@]}" out=$OUT/wclass5.tsv \
  wclasses=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5 > $OUT/wclass5.log 2>&1 &
wait
touch $OUT/done
