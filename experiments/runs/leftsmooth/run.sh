#!/bin/zsh
# Evidence-side leftover smoothing (leftsmooth=evidence): o(x|S∪E) = o(x|S) * P(E|x,S) / P(E|¬x,S), each
# rate from E's odds among the chosen / not-chosen records of S smoothed toward E's odds in S (the
# user's formulation), instead of smoothing o(x|S∪E) toward o(x|S).  Soft pool model (sigma 0.1, drop +
# keepgen), gate 0.5 + relevance 0.25, redundant, rootabsent; slice ceiling, Mac, 6 workers.
# Compare: soft pool 0.7478 / 0.7044; + product domains (eps 0) + greedyby=odds 0.7813 / 0.7389.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/leftsmooth
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true leftsmooth=evidence)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR\|recorded events' $OUT/$name.log | grep -v '^--'; }
{
run evidence
run evidence-odds-product0 greedyby=odds domains=product domeps=0
run evidence-odds greedyby=odds
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
