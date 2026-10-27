#!/bin/zsh
# Beta shrinkage of every chain and leftover estimate toward its parent's (shrink=k pseudo-records: 5, 20, 50),
# instead of the one-count smoothing; and distback (distance -> direction in leftover ladders), rerun after the
# ChainOpts field-order fix (the earlier run had it off).  Best ceiling setup: soft pool + greedyby=odds +
# leftsmooth=evidence + accum + projective + product domains (eps 0) + head_trank: 0.8531 / 0.8020.
# Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/shrink
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -E '^ALL|^score|ERROR' $OUT/$name.log; }
{
run shrink20 shrink=20
run distback distback=true
run shrink5 shrink=5
run shrink50 shrink=50
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
