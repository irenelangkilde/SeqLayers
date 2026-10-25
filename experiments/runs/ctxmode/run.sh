#!/bin/bash
# Leftover contexts that generalize or cover the chain (ctxmode): backoff (start from the chain; while
# chain'+leftover is unsupported, generalize the chain feature -- form -> xpos/upos/basepos(/classes)
# -> dropped -- that changes the chain's odds least), limited (no dropping: unsupported leftovers
# rejected), cover (greedy growth from chain + pool + the chain forms' generalizations; rejected
# unless it covers every chain feature).  Base: soft pool (sigma 0.1, drop + keepgen) + greedyby=odds
# + leftsmooth=evidence + product domains (eps 0): 0.7844 / 0.7406 (Mac); gate 0.5 + relevance 0.25,
# redundant, rootabsent; slice ceiling, two runs at a time, 6 workers each.  
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/ctxmode
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence domains=product domeps=0 )
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|recorded events\|Elapsed (wall\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{
  run baseline
  run backoff ctxmode=backoff
} > $OUT/part2.txt 2>&1 &
{
  run limited ctxmode=limited
  run cover ctxmode=cover
} > $OUT/part4.txt 2>&1 &
wait
cat $OUT/part*.txt > $OUT/summary.txt
touch $OUT/done
