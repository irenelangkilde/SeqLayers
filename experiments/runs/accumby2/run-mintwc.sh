#!/bin/zsh
# accumby=mintwc on base B: mintw, then past the first supported context keep backing off by the ΔW move while W's
# estimate stays within tau (mindevtau 0.1, 0.25) of its first-supported value.  mintw 0.8482 / 0.7954, odds 0.8531.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/accumby2
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true accumby=mintwc)
{
for t in 0.1 0.25; do
  echo "=== mintwc tau $t  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" mindevtau=$t note=accumby-mintwc-$t > $OUT/accumby-mintwc-$t.log 2>&1
  grep -E '^ALL|^double|^mintw|^score|ERROR' $OUT/accumby-mintwc-$t.log
done
echo "=== all done $(date)"
} > $OUT/summary-mintwc.txt 2>&1
