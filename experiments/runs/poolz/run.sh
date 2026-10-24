#!/bin/zsh
# Soft pool distance in standard errors (poolscale=z): d_g / SE, SE = sqrt(1/n_chosen + 1/n_not) at the
# chain's final prefix, weight exp(-z/sigma); sigma 1 and 2 (SE units).  Base: drop + keepgen soft pool
# (0.7478 / 0.7044, 941 s), gate 0.5 + relevance 0.25, redundant, rootabsent; slice, warm, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/poolz
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true
  poolmarginal=drop poolkeepgen=true poolscale=z)
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR\|recorded events\|using cached' $OUT/$name.log | grep -v '^--'; }
{
run z1 poolsigma=1.0
run z2 poolsigma=2.0
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
