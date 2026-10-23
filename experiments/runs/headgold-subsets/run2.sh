#!/bin/bash
# The missing gold subsets: XPOS alone, UPOS alone, distance + XPOS.  Same base and events as
# run.sh (gate 0.5 + relevance 0.25, rootabsent, noise=20).  Waits for runs/sdist.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/headgold-subsets
until [ -f experiments/runs/sdist/done ]; do sleep 60; done
# src changed after these events were recorded (options that leave them unchanged when off): refresh the stamp
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|^pipeline" $OUT/$name.log | grep -v "^--"; }
{
run hg-xpos      headgold=head_xpos
run hg-upos      headgold=head_upos
run hg-dist-xpos headgold=head_dist,head_xpos
echo "=== all done $(date)"
} > $OUT/summary2.txt 2>&1
touch $OUT/done2
