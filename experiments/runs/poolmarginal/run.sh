#!/bin/zsh
# Marginal pool filter (poolmarginal): the redundant pool admits only features informative on their own
# (|log o(c|r) - log o(c)| >= tau) that the chain makes redundant (< tau given the chain); features weak
# both ways stay leftovers and face the relevance filter.  Base: gate 0.5 + relevance 0.25 + redundant
# pool, rootabsent; slice, warm, Mac.  Compare: soft pool whole sigma 0.1 0.7492 / 0.7058 (runs/poolsigma),
# hard pool 0.7413 / 0.6999 (runs/redundant).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/poolmarginal
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolmarginal=true)
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR\|recorded events\|using cached' $OUT/$name.log | grep -v '^--'; }
{
run soft01-marginal poolsigma=0.1
run hard-marginal
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
