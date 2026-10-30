#!/bin/zsh
# Stick-breaking candidate probabilities (user's idea 2026-10-30) on base B (0.8531 / 0.8020):
#   renorm  values in rank order, p(r_k) = q_k * (what r_1..r_{k-1} leave), renormalized (argmax unchanged; changes the
#           product's attachment and head-property terms)
#   cond    labeled decisions: each later value's odds first conditioned on not the earlier values (its not-chosen
#           records less their chosen records, in its own final prefix); attachment as renorm
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/stickbreak
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
{
for sb in renorm cond; do
  echo "=== stickbreak=$sb  $(date)"
  julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 \
    headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
    poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 \
    headtrank=true stickbreak=$sb note=stickbreak-$sb > $OUT/stickbreak-$sb.log 2>&1
  grep -E '^ALL|^score|ERROR' $OUT/stickbreak-$sb.log
done
} > $OUT/summary.txt 2>&1
