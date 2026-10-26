#!/bin/bash
# Early confident head-property commits: two fixes compared, full decodes (slice), accum + projective setup.
#  1. DFS + backjumping with head-property choice points down to 0.5% (dfsepshead=0.005; others 5%)
#  2. soft head properties (softheads): never committed before their attachment; they weight its candidates
#     (product) and are derived from it -- greedy, and with DFS + backjumping
# Compare (same setup): greedy 0.6766 / 0.5999, plain DFS 0.6770 / 0.5999.  The first run records the events.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/softheads
COMMON=(model=chain select=maxdev events=gold dev_sents=5 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25
  gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence ctxmode=accum projective=true)
DFS=(dfs=true dfseps=0.05 dfsbudget=200 dfsjump=true)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/dev_eval.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^dfs\|^build\|ERROR\|recorded events\|Elapsed (wall\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
run dfs-jump-epshead workers=12 "${DFS[@]}" dfsepshead=0.005 > $OUT/part0.txt 2>&1
{ run softheads-greedy workers=6 softheads=true; } > $OUT/part1.txt 2>&1 &
{ run softheads-dfs-jump workers=6 softheads=true "${DFS[@]}"; } > $OUT/part2.txt 2>&1 &
wait
cat $OUT/part0.txt $OUT/part1.txt $OUT/part2.txt > $OUT/summary.txt
touch $OUT/done
