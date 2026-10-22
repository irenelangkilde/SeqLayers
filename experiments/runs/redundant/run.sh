#!/bin/zsh
# The redundant pool alone: features supported at the chain's final prefix but changing it by less
# than tau join the greedy leftovers' conditioning pool (no correction of their own), so a 1-feature
# chain can still reach minctx=2 contexts.  No chain-sized contexts.  Base: gate 0.5 + relevance
# 0.25, rootabsent; slice, warm.  Compare with gate05-rel (UAS 0.7399 / LAS 0.6985).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/redundant
# src changed after these events were recorded (options that leave them unchanged when off): refresh the stamp
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
/usr/bin/time -l julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true \
  redundant=true note=redundant > $OUT/run.log 2>&1
touch $OUT/done
