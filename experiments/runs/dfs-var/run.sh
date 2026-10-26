#!/bin/bash
# DFS variants in the full decode (slice): conflict-directed backjumping (dfsjump), product value ordering
# (dfsorder=product), both.  Setup: soft pool + greedyby=odds + leftsmooth=evidence + ctxmode=accum +
# projective; eps 0.05, budget 200.  Compare (Mac): greedy 0.6766 / 0.5999, plain DFS 0.6770 / 0.5999
# (21 of 25 segments exhausted the budget).  One after the other, 12 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/dfs-var
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=12 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25
  gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence ctxmode=accum projective=true dfs=true dfseps=0.05 dfsbudget=200)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/dev_eval.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^dfs\|^build\|ERROR\|Elapsed (wall\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{
run dfs-jump dfsjump=true
run dfs-product dfsorder=product
run dfs-jump-product dfsjump=true dfsorder=product
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
