#!/bin/bash
# Pair steps (synergy) with relations + projective and the current best setup: soft pool + greedyby=odds +
# leftsmooth=evidence + ctxmode=accum + product domains (eps 0); a chosen pair's members enter the chain in
# slot order (as before) or the one with the larger single effect first (pairorder=maxdev).  Both at once,
# 2 workers each.  Compare: without pairs 0.8769 / 0.8179.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/pairs-relproj
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs-rel
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=2 maxcached=50000000 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 relations=true projective=true
  redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum
  domains=product domeps=0 pairsteps=true pairscore=synergy)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{ run relproj-pairs-slot pairorder=slot; } > $OUT/slot.txt 2>&1 &
{ run relproj-pairs-maxdev pairorder=maxdev; } > $OUT/maxdev.txt 2>&1 &
wait
cat $OUT/slot.txt $OUT/maxdev.txt > $OUT/summary.txt
touch $OUT/done
