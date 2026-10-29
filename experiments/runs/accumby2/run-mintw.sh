#!/bin/zsh
# accumby=mintw (user's design 2026-10-29) on base B (odds 0.8531 / 0.8020): per backoff step, the ΔT proposal (odds
# rule) against the ΔW proposal (smallest |log o(x|W,C) - log o(x|W,C')| where W has records in both), the smaller
# change taken.  Mac, 6 workers, dcstats.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/accumby2
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 \
  headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
  poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 \
  headtrank=true dcstats=true accumby=mintw note=accumby-mintw > $OUT/accumby-mintw.log 2>&1
grep -E '^ALL|^double|^mintw|^score|ERROR' $OUT/accumby-mintw.log > $OUT/summary-mintw.txt
