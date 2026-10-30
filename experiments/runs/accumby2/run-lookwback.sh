#!/bin/zsh
# Pairwise backoff for the interaction (2026-10-30): where W is thin, |corr(W|a) - corr(W|a')| over all records
#   look1int + lookwsmooth=5   interaction from W's likelihood ratio, rates smoothed by 5 records toward W's pooled rate
#   look1red + lookw=min       supported moves by min(ΔW, interaction)
#   look1red + lookw=max       by max(ΔW, interaction): a move is safe only if both are unchanged
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/accumby2
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
{
for v in "accumby=look1int lookwback=5" "accumby=look1int lookwback=5 lookwsoft=true" "accumby=look1int lookwback=20 lookwsoft=true"; do
  n=$(echo $v | tr ' ' '_' | tr '=' '-')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 \
    headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
    poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 \
    headtrank=true dcstats=true ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^double|^lookwback|^score|ERROR' $OUT/$n.log
done
} > $OUT/summary-lookwback.txt 2>&1
