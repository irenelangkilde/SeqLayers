#!/bin/bash
# Which head properties matter: the dependent's head properties given at their gold values (subsets),
# then narrowing, inference and the attachment choice.  Base: gate 0.5 + relevance 0.25, rootabsent,
# noise=20; slice, warm.  Nothing given: UAS 0.7399 / LAS 0.6992 (gate05-rel-noise).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/headgold-subsets
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|^pipeline" $OUT/$name.log | grep -v "^--"; }
{
run hg-dir      headgold=head_dir
run hg-dist     headgold=head_dist
run hg-dirdist  headgold=head_dir,head_dist
run hg-dirdist-upos headgold=head_dir,head_dist,head_upos
run hg-dirdist-xpos headgold=head_dir,head_dist,head_xpos
run hg-all      headgold=all
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
