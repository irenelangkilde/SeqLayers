#!/bin/zsh
# Thin contexts in the leftovers, chain steps untouched: Beta shrinkage of the leftovers' corrections only
# (shrinkleft = 20, 50) and two-sided support (leftmin2 = 2, 5: context + leftover needs that many chosen and that
# many not-chosen records).  Best ceiling setup (head_trank): 0.8531 / 0.8020.  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/leftshrink
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -E '^ALL|^score|ERROR' $OUT/$name.log; }
{
run shrinkleft20 shrinkleft=20
run leftmin2-2 leftmin2=2
run shrinkleft50 shrinkleft=50
run leftmin2-5 leftmin2=5
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
