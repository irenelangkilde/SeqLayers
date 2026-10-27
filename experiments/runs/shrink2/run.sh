#!/bin/zsh
# Beta shrinkage rerun after clamping it to valid probabilities (sampled counts could make p >= 1: infinite / NaN
# odds, candidates lost: head_lost 4-15% in runs/shrink).  k = 20, 50, 200.  Best ceiling setup (head_trank):
# 0.8531 / 0.8020, head_lost 1.6%.  Slice ceiling, Mac, 6 workers.  Waits for runs/headwc.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/shrink2
until [ -f experiments/runs/headwc/done ]; do sleep 60; done
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -E '^ALL|^score|ERROR|^domains' $OUT/$name.log; }
{
run shrink50 shrink=50
run shrink200 shrink=200
run shrink20 shrink=20
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
