#!/bin/zsh
# Calibrated leftover weights (leftalpha) on the current accum setup: soft pool + greedyby=odds +
# leftsmooth=evidence + ctxmode=accum + projective.  1. fit alpha per decision type on the held-out tuning
# sentences (ceiling.jl on=tune calibrate=true; domains off: the calibration reads the attachment scores the
# product then uses); 2. the slice ceiling with the fitted alpha and product domains (eps 0), vs 0.8465 / 0.7927.
# Mac, 4 workers.  Waits for the DFS smoke test.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/accum-alpha
until grep -qs "^ALL\|ERROR" experiments/runs/dfs-smoke/dfs.log; do sleep 60; done
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=4 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true)
{
echo "=== calibrate  $(date)"
julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" on=tune calibrate=true note=accum-calibrate > $OUT/calibrate.log 2>&1
sed -n "/^calibration:/,/^alpha table/p" $OUT/calibrate.log
grep -E "ERROR" $OUT/calibrate.log
A=$W/events/alpha.tsv
if [ -f $A ]; then
  cp $A $OUT/alpha.tsv
  echo "=== leftalpha  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" domains=product domeps=0 leftalpha=$OUT/alpha.tsv note=accum-leftalpha > $OUT/leftalpha.log 2>&1
  grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^score|ERROR" $OUT/leftalpha.log
else
  echo "no alpha table"
fi
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
