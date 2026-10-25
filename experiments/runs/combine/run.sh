#!/bin/zsh
# Combining the two gains on the soft pool model (sigma 0.1, drop + keepgen; gate 0.5 + relevance 0.25,
# redundant, rootabsent): product domains with no pruning (domains=product domeps=0; 0.7779 / 0.7316 on
# the noise-20 events) and greedyby=odds leftover contexts (0.7606 / 0.7168), plus ctxfromchain.
# Slice ceiling, Mac, 6 workers.  Product alone first, on these (rootabsent) events.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/combine
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true domains=product domeps=0)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR\|recorded events\|^domains' $OUT/$name.log | grep -v '^--'; }
{
run product0
run product0-odds greedyby=odds
run product0-odds-fromchain greedyby=odds ctxfromchain=true
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
