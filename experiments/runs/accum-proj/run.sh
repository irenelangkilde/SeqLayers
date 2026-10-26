#!/bin/zsh
# Projectivity (candidate filter: no arc crossing a committed arc; events unchanged) on the current best
# cheap setup: soft pool + greedyby=odds + leftsmooth=evidence + ctxmode=accum, with product domains (eps 0;
# 0.7937 / 0.7485 without projectivity) and without them.  Old base + projective: 0.8213 / 0.7647
# (from 0.7399).  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/accum-proj
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -E '^ALL|^score|ERROR|^pipeline' $OUT/$name.log; }
{
run proj-product0 domains=product domeps=0
run proj-nodomains
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
