#!/bin/bash
# AWS part of runs/accumby2: one accumby variant (args), 8 workers; marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
TAG=$1; shift
OUT=experiments/runs/accumby2
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=8 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
n=$(echo "$*" | tr ' ,' '_+' | tr '=' '-')   # (GNU tr reads '_-+' as a range)
{
  echo "=== $*  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^double|^score|ERROR' $OUT/$n.log
  echo "=== $TAG done $(date)"
} > $OUT/summary-$TAG.txt 2>&1
touch $OUT/done-$TAG
