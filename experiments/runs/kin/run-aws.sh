#!/bin/bash
# Neighbour kinship (user's design 2026-11-01): the token's kinship to its left / right neighbour (parent, dep, sibling,
# aunt/uncle, niece/nephew, 1st / 2nd cousin, grandparent, other ancestor / descendant / relative, none) as attachment
# items (kinitems), as head-property decisions in the product (headkin=left,right), or both; on base B (0.8531 / 0.8020)
# and base B + idxeq (0.8562 / 0.8051).  New specs: each variant records its own events.  AWS, three at a time,
# 5 workers each; marks done-aws.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/kin
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
one() {
  n=kin_$(echo "$*" | tr ' ,' '_+' | tr '=' '-')   # (GNU tr reads '_-+' as a range)
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$n > $OUT/$n.log 2>&1
  { echo "=== $*"; grep -E '^ALL|^score|ERROR' $OUT/$n.log; grep -E 'head_kin' $OUT/$n.log | head -4; } >> $OUT/summary-aws.txt
}
echo "=== start $(date)" > $OUT/summary-aws.txt
one kinitems=true & sleep 30
one headkin=left,right & sleep 30
one kinitems=true headkin=left,right &
wait
one idxeq=true kinitems=true & sleep 30
one idxeq=true headkin=left,right & sleep 30
one idxeq=true kinitems=true headkin=left,right &
wait
echo "=== all done $(date)" >> $OUT/summary-aws.txt
touch $OUT/done-aws
