#!/bin/zsh
# Target mode (top-two restarts): chains run for all candidates in lockstep; when the identity of
# the top two changes during selection, restart with those odds as targets (maxrestarts 2,
# immediate).  On the current base: gate 0.5 + relevance 0.25, rootabsent; slice.  Waits for
# runs/redctx.  Compare with gate05-rel (independent maxdev chains).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/target
until [ -f experiments/runs/redctx/done ]; do sleep 60; done
# src changed after these events were recorded (chain.jl only, not recording): refresh the stamp
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
/usr/bin/time -l julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true \
  target=true maxrestarts=2 restart=immediate note=target > $OUT/run.log 2>&1
touch $OUT/done
