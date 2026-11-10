#!/bin/bash
# Delayed relevance filter (relstrikes; user's design 2026-11-10): no up-front relevance filter on the leftovers; a
# leftover under tau (or unsupported) in a round gets a strike and leaves at relstrikes -- dropped, or pooled into the
# context (relpool).  On base B + kin + nn2 (0.8782 / 0.8230).  Args: tag, then configs as name=options (options
# comma-separated), two at a time, 7 workers each.  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/relstrikes
TAG=$1; shift
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true
  neighbor2=true kinitems=true headkin=left,right)
one() {
  n=relstrikes_${1%%=*}
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=7 $(echo "${1#*=}" | tr ',' ' ' | tr '+' ',') note=$n > $OUT/$n.log 2>&1
  { echo "=== ${1%%=*}"; grep -E '^ALL|^score|^forkcap|recorded|ERROR' $OUT/$n.log; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
while [ $# -gt 0 ]; do
  one "$1" & sleep 30
  [ -n "$2" ] && one "$2" &
  wait; shift; [ $# -gt 0 ] && shift
done
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
