#!/bin/bash
# Do the leftover-redundancy fixes lose because of the product domains?  The same variants with domains=off (no
# head-property product terms), each against a no-domains control.  Base otherwise: best ceiling setup (head_trank;
# with product domains 0.8531 / 0.8020).  Variants with domains (same day): telescope 0.8451 / 0.7885, depweight
# 0.8403 / 0.7858, accumby=dep 0.8458 / 0.7948, dep+depweight 0.8413 / 0.7872, ciorder=value 0.8441 / 0.7937.
# Slice ceiling, AWS (16 cores), 8 workers.  Usage: run.sh A | B.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/nodomains
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=8 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=off headtrank=true)
if [ "$1" = A ]; then
  RUNS=("note=control" "telescope=true" "depweight=true")
else
  RUNS=("accumby=dep" "accumby=dep depweight=true" "ciorder=value")
fi
{
for v in "${RUNS[@]}"; do
  n=$(echo $v | tr ' =' '_-')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" $v note=nodom-$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary-$1.txt 2>&1
touch $OUT/done-$1
