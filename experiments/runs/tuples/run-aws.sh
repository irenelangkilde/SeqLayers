#!/bin/bash
# Tuple first step (user's design 2026-10-31) on base B (0.8531 / 0.8020): each maxdev chain's first step adds the best
# supported triple (else pair, else single) of its pool by max_x l(X = x | tuple); then singleton steps and base B's
# leftovers (greedy, greedyby=odds, leftsmooth=evidence, ctxmode=accum, accumby=odds).  Pools: tuplek=6, tuplemin=1.0,
# tuplek=4.  AWS, three runs at once, 5 workers each; marks done-aws.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/tuples
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
one() {
  n=tuple_$(echo "$1" | tr '=' '-')
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" $1 note=$n > $OUT/$n.log 2>&1
  { echo "=== $1"; grep -E '^ALL|^tuple|^score|ERROR' $OUT/$n.log; } >> $OUT/summary-aws.txt
}
echo "=== start $(date)" > $OUT/summary-aws.txt
one tuplek=6 & sleep 20; one tuplemin=1.0 & sleep 20; one tuplek=4 &
wait
echo "=== all done $(date)" >> $OUT/summary-aws.txt
touch $OUT/done-aws
