#!/bin/zsh
# Freeze-and-switch (lockstep=top2: only the two leading candidates' chains grow; a chain freezes when
# its candidate drops to rank 3 or lower, and the newcomer resumes its own) with gated, filtered greedy
# leftovers and the redundant pool (so short frozen chains can still reach the 2-feature minimum).
# Base: gate 0.5 + relevance 0.25, rootabsent; slice, warm.  Waits for runs/adaptive.  Compare with
# runs/redundant (the same without freezing: 0.7413 / 0.6999).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/top2-redundant
until [ -f experiments/runs/adaptive/done ]; do sleep 60; done
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"   # options only changed in src
/usr/bin/time -l julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true \
  redundant=true lockstep=top2 note=top2-redundant > $OUT/run.log 2>&1
touch $OUT/done
