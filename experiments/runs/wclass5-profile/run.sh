#!/bin/zsh
# Where does the time go with word classes?  One process, the 86-token profile doc, cold (no count
# cache), with flat profile.  Compare with runs/greedy-rel025-mmap/profile-countcache-false.log (224 s).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/wclass5-profile
/usr/bin/time -l julia --project=experiments experiments/profile_chain.jl model=chain select=maxdev events=gold dev_sents=5 \
  jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 countcache=false flat=300 \
  wclasses=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5 > $OUT/profile.log 2>&1
touch $OUT/done
