#!/bin/zsh
# Contest re-ranking: after the usual scores (independent maxdev chains + gated, filtered greedy
# leftovers), the top two are re-decided by a contest chain on their odds ratio (from zero features,
# no leftovers); a new contest when one drops below a third (maxrestarts 2).  Base: gate 0.5 +
# relevance 0.25, rootabsent; slice, warm.  Compare with gate05-rel (UAS 0.7399 / LAS 0.6985).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/contest
# src changed after these events were recorded (chain.jl and harness options only): refresh the stamp
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
/usr/bin/time -l julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true \
  contest=true maxrestarts=2 note=contest > $OUT/run.log 2>&1
touch $OUT/done
