#!/bin/zsh
# Mac part of runs/accumby2: accumby=marg (already running as PID 41207, log accumby-marg.log), then accumby=mindev.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/accumby2
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
{
echo "=== accumby=marg  (started 09:28:59)"
while kill -0 41207 2>/dev/null; do sleep 30; done
grep -E '^ALL|^double|^score|ERROR' $OUT/accumby-marg.log
for v in "accumby=mindev"; do
  n=$(echo $v | tr ' =,' '_-+')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^double|^score|ERROR' $OUT/$n.log
done
echo "=== mac done $(date)"
} > $OUT/summary-mac.txt 2>&1
touch $OUT/done-mac
