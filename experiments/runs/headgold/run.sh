#!/bin/zsh
# Ceiling for the attachment mechanism: the word's head properties given at their gold values
# (all four; then direction and distance only), then narrowing, inference, and the attachment
# choice among what's left.  Best config (greedy, relevance 0.25), no classes, slice, warm cache.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/headgold
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 countmin=10000)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR\|^pipeline' $OUT/$name.log | grep -v '^--'; }
{
run headgold-all headgold=all
run headgold-dirdist headgold=head_dir,head_dist
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
