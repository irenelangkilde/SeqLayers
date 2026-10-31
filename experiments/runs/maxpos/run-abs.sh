#!/bin/zsh
# One shared chain per decision (select=maxabsscore; 2026-10-31): each step the feature growing some value's |log odds| most
# (either sign; negatives fine); stops when none grows it by tau.  On base B + idxeq (maxdev 0.8562 / 0.8051).
# Waits for the maxpos run.  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
while pgrep -f "[n]ote=maxpos-idxeq" > /dev/null; do sleep 30; done
OUT=experiments/runs/maxpos
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank-idxeq
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
julia --project=experiments experiments/ceiling.jl model=chain select=maxabsscore events=gold dev_sents=5 workers=6 tags=false jointmin=10000 \
  headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
  poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 \
  headtrank=true idxeq=true note=maxabsscore-idxeq > $OUT/maxabsscore-idxeq.log 2>&1
grep -E '^ALL|^score|ERROR' $OUT/maxabsscore-idxeq.log > $OUT/summary-abs.txt
