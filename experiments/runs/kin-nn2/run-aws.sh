#!/bin/bash
# neighbor2 + kinitems + headkin=left,right together (user 2026-11-01), on base B (0.8531 / 0.8020; best singles: both
# kin 0.8727 / 0.8161, neighbor2 0.8675 / 0.8148) and base B + idxeq (0.8562 / 0.8051; 0.8741 / 0.8165, 0.8679 / 0.8141).
# New specs: each records its own events.  AWS, both at once, 7 workers each; marks done-aws.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/kin-nn2
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=7 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
one() {
  n=kinnn2_$(echo "$*" | tr ' ,' '_+' | tr '=' '-')   # (GNU tr reads '_-+' as a range)
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$n > $OUT/$n.log 2>&1
  { echo "=== $*"; grep -E '^ALL|^score|ERROR' $OUT/$n.log; grep -E 'head_kin' $OUT/$n.log | head -4; } >> $OUT/summary-aws.txt
}
echo "=== start $(date)" > $OUT/summary-aws.txt
one neighbor2=true kinitems=true headkin=left,right & sleep 30
one idxeq=true neighbor2=true kinitems=true headkin=left,right &
wait
echo "=== all done $(date)" >> $OUT/summary-aws.txt
touch $OUT/done-aws
