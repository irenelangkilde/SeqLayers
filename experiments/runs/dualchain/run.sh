#!/bin/zsh
# Dual chains (user's idea 2026-10-30): per candidate a positive chain P and a negative chain N; a feature whose sign
# turns when added joins the other chain too.  Eligibility: marginal (by marginal sign) | inchain (by in-chain sign).
# Combine: sum | maxabs | weighted (N x 0.5) | sumdedup (sum without duplicates).  Leftovers on (context P ∪ N) / off.
# 16 runs on base B (0.8531 / 0.8020).  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/dualchain
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
{
for el in marginal inchain; do
  for cb in sum maxabs weighted sumdedup; do
    for dl in true false; do
      n=dual_${el}_${cb}_left-$dl
      echo "=== $el $cb leftovers=$dl  $(date)"
      julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 \
        headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
        poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 \
        headtrank=true dualchain=$el dualcombine=$cb dualleft=$dl note=$n > $OUT/$n.log 2>&1
      grep -E '^ALL|^score|ERROR' $OUT/$n.log
    done
  done
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
