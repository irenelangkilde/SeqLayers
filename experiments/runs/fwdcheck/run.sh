#!/bin/bash
# Forward checking (fwdcheck): once no other head can appear, a head-property value is a candidate only if some
# possible head has it (a head property left with none is a DFS failure).  Full decodes (slice): greedy, and DFS
# with backjumping.  Same setup as runs/dfs-var (accum + projective; eps 0.05, budget 200); 12 workers.
# Waits for runs/dfs-var.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/fwdcheck
until [ -f experiments/runs/dfs-var/done ]; do sleep 60; done
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=12 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25
  gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence ctxmode=accum projective=true fwdcheck=true)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/dev_eval.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^dfs\|^build\|ERROR\|Elapsed (wall\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{
run fwd-greedy
run fwd-dfs-jump dfs=true dfseps=0.05 dfsbudget=200 dfsjump=true
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
