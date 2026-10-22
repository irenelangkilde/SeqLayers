#!/bin/zsh
# Redundant features in the leftovers' conditioning pool, leftover contexts exactly as long as the
# chain; on gate 0.5 + relevance 0.25, rootabsent, no classes; slice.  Waits for runs/gate05-rel.
# Compare with gate05-rel and with runs/redctx-wc (the same with clusters as features).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/redctx
until [ -f experiments/runs/gate05-rel/done ]; do sleep 60; done
# src changed after these events were recorded (chain.jl and dev_eval.jl options only, which don't
# affect recording): refresh the stamp so the events and count cache are reused
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
/usr/bin/time -l julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true \
  redundant=true ctxsize=chain note=redctx > $OUT/run.log 2>&1
touch $OUT/done
