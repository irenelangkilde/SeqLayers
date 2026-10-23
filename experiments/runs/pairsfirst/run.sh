#!/bin/zsh
# Pairs first: with pair steps (synergy), features in any precomputed pair join a chain only via a
# pair.  First pairs-synergy itself on these events (Mac, no noise feature) for a same-machine
# comparison, then pairsfirst.  Base: gate 0.5 + relevance 0.25 + redundant pool, rootabsent; slice.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/pairsfirst
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true pairsteps=true pairscore=synergy)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -l julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR' $OUT/$name.log | grep -v '^--'; }
{
run pairs-synergy-mac
run pairsfirst pairsfirst=true
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
