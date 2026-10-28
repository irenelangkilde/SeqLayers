#!/bin/zsh
# accumreset=true (each leftover's backoff restarts from the chain + earlier leftovers at their most specific;
# its generalizations aren't carried forward) on the recent accum variants.  Base B (accum + projective + product
# + head_trank) without reset: base 0.8531 / 0.8020, depodds 0.8513, dep (Yule) 0.8458, dep pmi 0.8448,
# telescope 0.8451, branches=3 mix 0.8468, accumwin2 (base A) 0.8403, treerank+headtrank 0.8534 / 0.8010.
# Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/resetvar
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
W2=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-trank-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); s = srcstamp(); write(\"$W/stamp\", s); write(\"$W2/stamp\", s)"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true accumreset=true)
{
for v in "base" "accumby=depodds" "accumby=dep" "accumby=dep depmeasure=pmi" "telescope=true" "branches=3 combine=mix" \
  "accumwin=2" "treerank=true"; do
  n=reset_$(echo $v | tr ' =,' '_-+')
  [[ $v == base ]] && v=""
  echo "=== reset $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
