#!/bin/zsh
# Relative PP items (pprel): each PP item compares the candidate's DepCC rate for the dependent's
# preposition with the best rate among the token's other candidates (log2 ratio buckets) instead of
# bucketing the rate alone.  1. base + pprel (records new events); 2. + pair steps (synergy).
# Base: gate 0.5 + relevance 0.25 + redundant pool, rootabsent; slice, Mac, 6 workers.
# Compare: absolute pptable 0.7465 / 0.7040, with pairs 0.7706 / 0.7292 (runs/pptable);
# no pptable 0.7413 / 0.6999, pairs-synergy 0.7703 / 0.7296.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/pprel
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true
  pptable=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/pptable/pptable.tsv pprel=true)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR\|recorded events\|using cached' $OUT/$name.log | grep -v '^--'; }
{
run pprel-base
run pprel-pairs pairsteps=true pairscore=synergy
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
