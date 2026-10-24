#!/bin/zsh
# Drop weak non-redundant features (poolmarginal=drop): features < tau given the chain that are also < tau
# on their own are discarded (not pooled, not leftovers).  1. drop all of them; 2. keep generalizations
# of chain forms in the pool (poolkeepgen; the soft pool weighs those).  Base: soft pool whole sigma 0.1
# + redundant pool, gate 0.5 + relevance 0.25, rootabsent; slice, warm, Mac, 6 workers.
# Compare: soft pool 0.7492 / 0.7058 (1850 s); weak ones as leftovers 0.7447 / 0.7009 (4334 s).
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/pooldrop
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop)
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 '^ALL\|^score\|ERROR\|recorded events\|using cached' $OUT/$name.log | grep -v '^--'; }
{
run soft01-drop
run soft01-drop-keepgen poolkeepgen=true
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
