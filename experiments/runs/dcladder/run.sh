#!/bin/zsh
# Toward a base with the least double counting (dcstats=true: per accum leftover taken, the largest |Yule's Q| to a
# counted feature outside its context).  Base B (accum + projective + product + head_trank) 0.8531 / 0.8020:
# 69.7% of leftovers with |Q| >= 0.5, |corr|-weighted 0.687 (smoke run).  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/dcladder
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
{
for v in "accumreset=true" "telescope=true" "telescope=true accumreset=true" "telescope=true accumreset=true depweight=true" \
  "telescope=true accumreset=true depweight=true prodprops=head_xpos,head_dir,head_dist"; do
  n=dc_$(echo $v | tr ' =,' '_-+')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^double|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
