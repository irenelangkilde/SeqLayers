#!/bin/bash
# Domain variables, follow-ups: 1. product scoring with smaller eps (1e-4, 1e-5) and no pruning (0) on
# the soft pool model (product 1e-3: 0.7754 / 0.7296; baseline 0.7478 / 0.7051), 6 workers;
# 2. alongside, relations + projective (0.8706 / 0.8092) with product domains at 1e-3 and 1e-4,
# 2 workers.  (src changed since the events were recorded, events unchanged: stamps refreshed.)
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/domains2
for W in experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs-rel; do
  julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
done
SOFT=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 redundant=true
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true domains=product)
REL=(model=chain select=maxdev events=gold dev_sents=5 workers=2 maxcached=50000000 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 relations=true projective=true
  domains=product)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|^pipeline\|^domains\|Elapsed (wall\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{
  run soft-product-1e-4 "${SOFT[@]}" domeps=1e-4
  run soft-product-1e-5 "${SOFT[@]}" domeps=1e-5
  run soft-product-0 "${SOFT[@]}" domeps=0
} > $OUT/soft.txt 2>&1 &
{
  run relproj-product-1e-3 "${REL[@]}" domeps=1e-3
  run relproj-product-1e-4 "${REL[@]}" domeps=1e-4
} > $OUT/rel.txt 2>&1 &
wait
cat $OUT/soft.txt $OUT/rel.txt > $OUT/summary.txt
touch $OUT/done
