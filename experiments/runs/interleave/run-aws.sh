#!/bin/bash
# Interleaved chain + leftovers (user's design 2026-11-01): each step the best chain step competes with the best
# leftover (unsupported under the prefix so far, backed off as base B does, scored by its discounted correction); on
# the posneg-heads ceiling setup (control three_restartafter_posnegonly-heads 0.8578 / 0.8075).  Discounts: poolsigma=0.1
# (as base B), none (poolsigma=0), leftwback=1 + leftwovl max, accumby=depodds.  Args: tag, then one quoted option
# string per run; runs side by side, 7 workers each; marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/interleave
TAG=$1; shift
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=7 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true
  stepscore=posneg posnegrestart=true posnegafter=true posnegonly=heads interleave=true)
one() {
  n=il_$(echo "$1" | tr ' ,' '_+' | tr '=' '-')   # (GNU tr reads '_-+' as a range)
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" $1 note=$n > $OUT/$n.log 2>&1
  { echo "=== $1"; grep -E '^ALL|^interleave|^posnegrestart|^score|ERROR' $OUT/$n.log; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
first=1
for v in "$@"; do
  [ $first = 1 ] || sleep 30
  first=0
  one "$v" &
done
wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
