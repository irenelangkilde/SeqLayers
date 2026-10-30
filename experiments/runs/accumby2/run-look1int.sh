#!/bin/zsh
# accumby=look1red / look1ind (user's design 2026-10-29) on base B (odds 0.8531 / 0.8020; bfs 0.8499 / 0.7972): greedy,
# one level at a time -- a one-step backoff that supports W (>= 5 records) if any, by the smallest ΔW; otherwise back
# off the most redundant feature (min D(a→y), CI table; look1red) or the one most independent of W (min |Q|; look1ind).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/accumby2
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
{
for ab in look1int; do
  echo "=== accumby=$ab  $(date)"
  julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 \
    headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
    poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 \
    headtrank=true dcstats=true accumby=$ab note=accumby-$ab > $OUT/accumby-$ab.log 2>&1
  grep -E '^ALL|^double|^look1|^score|ERROR' $OUT/accumby-$ab.log
done
} > $OUT/summary-look1int.txt 2>&1
