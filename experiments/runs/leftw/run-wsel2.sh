#!/bin/zsh
# The rest of the wsel 2 x 2 (user's design 2026-10-31), N = 4: move by the weights alone vs by the largest |w * corr| among
# moves supporting the leftover (wselmove=wcorr); leftover correction unweighted vs times w (wselweight); w = min or (sum / 2)
# of w_back and w_ovl.  Base B (0.8531 / 0.8020).  Waits for the first wsel batch.  Mac, 6 workers, dcstats.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
until grep -q "all done" experiments/runs/leftw/summary-wsel.txt 2>/dev/null; do sleep 60; done
OUT=experiments/runs/leftw
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true
  accumby=wsel leftwback=4)
{
for cb in sum min; do
  for v in "wselmove=weights wselweight=true" "wselmove=wcorr wselweight=false" "wselmove=wcorr wselweight=true"; do
    n=wsel2_${cb}_$(echo $v | tr ' ' '_' | tr '=' '-')
    echo "=== wselcomb=$cb $v  $(date)"
    julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" wselcomb=$cb ${=v} note=$n > $OUT/$n.log 2>&1
    grep -E '^ALL|^double|^score|ERROR' $OUT/$n.log
  done
done
echo "=== all done $(date)"
} > $OUT/summary-wsel2.txt 2>&1
