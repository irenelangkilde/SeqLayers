#!/bin/zsh
# Pair steps (synergy) on the current best cheap setup + projective (no relations): soft pool + greedyby=odds +
# leftsmooth=evidence + ctxmode=accum + product domains (eps 0); pairorder slot vs maxdev.  Compare: without pairs
# 0.8465 / 0.7927.  Slice ceiling, Mac, 6 workers, one after the other.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/pairs-proj
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 pairsteps=true pairscore=synergy)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -E '^ALL|^score|ERROR' $OUT/$name.log; }
{
run proj-pairs-slot pairorder=slot
run proj-pairs-maxdev pairorder=maxdev
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
