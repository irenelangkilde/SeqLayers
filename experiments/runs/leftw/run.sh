#!/bin/zsh
# Accum leftover weights (user's design 2026-10-31) on base B (0.8531 / 0.8020): backoff w_back = exp(-(l_i - l_0)^2 / N)
# (N = 1, 4); overlap w_ovl = 1 - max overlap (Δe + Δa - Δea) / min(|Δe|, |Δa|) with counted features outside the context;
# both combined by product / min / max (N = 1).  Slice ceiling, Mac, 6 workers, dcstats.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/leftw
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
{
for v in "leftwback=1" "leftwback=4" "leftwovl=true" "leftwback=1 leftwovl=true leftwcomb=prod" \
  "leftwback=1 leftwovl=true leftwcomb=min" "leftwback=1 leftwovl=true leftwcomb=max"; do
  n=leftw_$(echo $v | tr ' ' '_' | tr '=' '-')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^double|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
