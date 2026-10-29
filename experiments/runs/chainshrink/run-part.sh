#!/bin/bash
# One machine's part of runs/chainshrink: args TAG WORKERS [WAITPID] then specs base:target:k (base B or DC).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
TAG=$1; NW=$2; WAITPID=$3; shift 3
OUT=experiments/runs/chainshrink
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=$NW tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
DC=(accumby=mindevsup explainpool=true depweight=true prodprops=head_xpos,head_dir,head_dist)
{
if [ "$WAITPID" != "-" ]; then
  while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
else
  julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
fi
for spec in "$@"; do
  IFS=: read base to k <<< "$spec"
  n=${base}_${to}_$k
  [ -f $OUT/$n.log ] && grep -q '^ALL' $OUT/$n.log && { echo "=== $base $to k$k (done before)"; grep -E '^ALL|^double|^score|ERROR' $OUT/$n.log; continue; }
  extra=(); [ $base = DC ] && extra=("${DC[@]}")
  echo "=== $base $to k$k  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" chainshrink=$k chainshrinkto=$to "${extra[@]}" note=chainshrink_$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^double|^score|ERROR' $OUT/$n.log
done
echo "=== $TAG done $(date)"
} > $OUT/summary-$TAG.txt 2>&1
touch $OUT/done-$TAG
