#!/bin/bash
# Accum + word classes (clusters/classes5): classes 30 and 300 added (wclevels=30,300) as ordinary features and
# as ladder rungs, alongside the POS tags and the rest; all four levels; and the control with no class level used
# (wclevels=1: same events, classes excluded).  Base: soft pool + greedyby=odds + leftsmooth=evidence +
# ctxmode=accum + projective + product domains (eps 0); without classes 0.8465 / 0.7927 (Mac).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/accum-wc
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-wcclasses5-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0
  wclasses=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{
  run wc-30-300 wclevels=30,300
  run wc-none wclevels=1
} > $OUT/part1.txt 2>&1 &
{
  run wc-all
} > $OUT/part2.txt 2>&1 &
wait
cat $OUT/part1.txt $OUT/part2.txt > $OUT/summary.txt
touch $OUT/done
