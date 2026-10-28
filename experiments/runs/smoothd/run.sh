#!/bin/bash
# Smoothing amount: smoothOdds moves an estimate d counts toward its parent's odds (default d = 1), now
# d = smoothd + smootha * n, n = the set's records (smoothn=set) or its parent's (smoothn=parent: chain steps and
# accum leftovers).  mincount=0 (support: one record) except the controls; relevance gates always built with
# mincount 5 (gatemincount=5) so the admitted slots don't change.  Base: best ceiling setup (head_trank) 0.8531 /
# 0.8020.  Slice ceiling, AWS (16 cores), 8 workers.  Usage: run.sh A | B  (the two machines' halves).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/smoothd
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=8 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true gatemincount=5)
if [ "$1" = A ]; then
  RUNS=("mincount=5" "mincount=0" "mincount=5 smoothd=5" "mincount=0 smoothd=5"
    "mincount=0 smoothd=0 smootha=0.05" "mincount=0 smoothd=0 smootha=0.1" "mincount=0 smoothd=0 smootha=0.2")
else
  RUNS=("mincount=0 smoothd=1 smootha=0.01" "mincount=0 smoothd=1 smootha=0.05" "mincount=0 smoothd=1 smootha=0.1"
    "mincount=0 smoothd=1 smootha=0.0001 smoothn=parent" "mincount=0 smoothd=1 smootha=0.001 smoothn=parent"
    "mincount=0 smoothd=0 smootha=0.001 smoothn=parent")
fi
{
for v in "${RUNS[@]}"; do
  n=$(echo $v | tr ' =' '_-')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" $v note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary-$1.txt 2>&1
touch $OUT/done-$1
