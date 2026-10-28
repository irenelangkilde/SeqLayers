#!/bin/zsh
# Accum leftovers: ladder telescoping (telescope=true: a leftover on a token's form/xpos/upos/basepos ladder waits for
# its pending coarser rungs, is left out under a finer chain rung, and is conditioned on its nearest counted coarser
# rung) and multiple contexts per leftover (branches=3: the first supported context along the paths starting with the
# 3 best first generalization steps; combine = avg (inverse variance) | ie (inclusion-exclusion over meets) | mix).
# Base: best ceiling setup (head_trank) 0.8531 / 0.8020 (control rerun 2026-10-27 identical).  Slice, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/telebranch
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true)
{
for v in "telescope=true" "branches=3 combine=avg" "branches=3 combine=ie" "branches=3 combine=mix" "telescope=true branches=3 combine=mix"; do
  n=$(echo $v | tr ' =' '_-')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
