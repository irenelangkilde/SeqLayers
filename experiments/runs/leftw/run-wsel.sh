#!/bin/zsh
# accumby=wsel (user's design 2026-10-31): each leftover its own backoff path, each move chosen by the largest min / sum of
# w_back = exp(-(l(C') - l_0)^2 / N) and w_ovl for the context it leads to; N = 1, 4.  Base B (0.8531 / 0.8020).
# Waits for the leftover-weight batch.  Slice ceiling, Mac, 6 workers, dcstats.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
until grep -q "all done" experiments/runs/leftw/summary.txt 2>/dev/null; do sleep 60; done
OUT=experiments/runs/leftw
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true accumby=wsel)
{
for v in "wselcomb=sum leftwback=1" "wselcomb=sum leftwback=4" "wselcomb=min leftwback=1" "wselcomb=min leftwback=4"; do
  n=wsel_$(echo $v | tr ' ' '_' | tr '=' '-')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^double|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary-wsel.txt 2>&1
