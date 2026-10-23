#!/bin/zsh
# The two missing three-of-four gold head-property subsets: direction + UPOS + XPOS (no distance) and
# distance + UPOS + XPOS (no direction).  Base: gate 0.5 + relevance 0.25, rootabsent (the AWS subsets
# also had noise=20, which the gates keep out of attachment).  Waits for runs/top2-redundant.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/headgold-triples
until [ -f experiments/runs/top2-redundant/done ]; do sleep 60; done
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR\|^pipeline' $OUT/$name.log | grep -v '^--'; }
{
run hg-dir-upos-xpos  headgold=head_dir,head_upos,head_xpos
run hg-dist-upos-xpos headgold=head_dist,head_upos,head_xpos
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
