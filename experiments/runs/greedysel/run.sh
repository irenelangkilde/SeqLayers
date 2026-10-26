#!/bin/bash
# Greedy (bottom-up) leftover contexts with the min-then-max rule: greedymin -- each leftover's context is the
# prefix of its greedy growth (>= minctx features) with the smallest |correction|; greedysel -- leftovers taken
# in rounds, the largest |correction| first, each joining the pool.  Base (best greedy-context setup): soft pool
# + greedyby=odds + leftsmooth=evidence + product domains eps 0, + projective (to match the accum runs); noise-20
# events.  baseline (neither), both, each alone.  Two at a time, 4 workers each.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/greedysel
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=4 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 redundant=true poolsigma=0.1 poolmarginal=drop
  poolkeepgen=true greedyby=odds leftsmooth=evidence projective=true domains=product domeps=0)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{
  run greedy-base
  run greedy-min-sel greedymin=true greedysel=true
} > $OUT/part1.txt 2>&1 &
{
  run greedy-sel greedysel=true
  run greedy-min greedymin=true
} > $OUT/part2.txt 2>&1 &
wait
cat $OUT/part1.txt $OUT/part2.txt > $OUT/summary.txt
touch $OUT/done
