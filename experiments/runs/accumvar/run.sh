#!/bin/bash
# Accum variants on the current cheap best setup (soft pool + greedyby=odds + leftsmooth=evidence + ctxmode=accum
# + projective + product domains eps 0), noise-20 events: 0.8468 / 0.7917 as is (accumwin 0, no restart).
#  accumwin=W: each leftover's context minimizes its |correction| over the first supported context and W further
#              generalization steps (2, 5, 1000 = the whole path)
#  accumreset: each leftover generalizes from the chain + earlier leftovers at their most specific; its
#              generalizations aren't carried forward
# Two at a time, 4 workers each.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/accumvar
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=4 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 redundant=true poolsigma=0.1 poolmarginal=drop
  poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{
  run accumwin2 accumwin=2
  run accumwin-all accumwin=1000
} > $OUT/part1.txt 2>&1 &
{
  run accumwin5 accumwin=5
  run accumreset accumreset=true
} > $OUT/part2.txt 2>&1 &
wait
cat $OUT/part1.txt $OUT/part2.txt > $OUT/summary.txt
touch $OUT/done
