#!/bin/bash
# Thin chain steps (user 2026-11-06; singleton chains led by a rare neighbour's form, 40 of 133 the gold head): stepmin=N
# (a chain step needs N records; thinner features stay leftovers) and forkthin=N (fork where the best step is thin: one
# branch takes it, the other the best step with N records; forkcap 16, forkcomb avg), N = 25 / 100 / 400.  On base B +
# kin + nn2 (0.8782 / 0.8230; fork-avg on lowering steps 0.8834 / 0.8337).  Pairs (at most two kin + nn2 ceilings
# building at once), 7 workers each.  Args: tag, stepmin | forkthin (one per machine).  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/thin
TAG=$1
KIND=$2
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true
  neighbor2=true kinitems=true headkin=left,right)
one() {
  n=thin_$2
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=7 $1 note=$n > $OUT/$n.log 2>&1
  { echo "=== $2"; grep -E '^ALL|^score|^forkcap|recorded|ERROR' $OUT/$n.log; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
arg() { [ $KIND = stepmin ] && echo "stepmin=$1" || echo "forkcap=16 forkcomb=avg forkthin=$1"; }
one "$(arg 25)" ${KIND}25 & sleep 30
one "$(arg 100)" ${KIND}100 &
wait
one "$(arg 400)" ${KIND}400
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
