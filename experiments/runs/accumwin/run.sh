#!/bin/zsh
# Accum leftovers whose context minimizes the leftover's |correction| (accumwin=W: among the first supported
# context and W further generalization steps), then the largest such correction is taken.  Current cheap best
# setup: soft pool + greedyby=odds + leftsmooth=evidence + ctxmode=accum + projective + product domains (eps 0):
# 0.8465 / 0.7927 (accumwin 0).  W = 2, 5, 1000 (the whole path).  Slice ceiling, Mac, 6 workers.
# Waits for the DFS smoke test.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/accumwin
until grep -qs "^ALL\|ERROR" experiments/runs/dfs-smoke/dfs.log; do sleep 60; done
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -E '^ALL|^score|ERROR' $OUT/$name.log; }
{
run accumwin2 accumwin=2
run accumwin5 accumwin=5
run accumwin-all accumwin=1000
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
