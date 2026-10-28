#!/bin/zsh
# Leftover-context selection by label-blind PMI between a candidate context item a and the leftover E:
# rho = log p(e,a) / p(e)p(a) (pmi) or max(rho, 0) (pmipos), ignoring x.  greedy: add the a with the largest rho
# (greedyby=dep); accum: the move losing the least rho (accumby=dep).  Yule's Q versions: greedy dep 0.8365,
# accum dep 0.8458.  Controls: greedy odds 0.8399 / 0.7879, accum odds 0.8531 / 0.8020.  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/pmictx
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true)
{
for v in "ctxmode=greedy greedyby=dep depmeasure=pmi" "ctxmode=greedy greedyby=dep depmeasure=pmipos" \
  "accumby=dep depmeasure=pmi" "accumby=dep depmeasure=pmipos"; do
  n=$(echo $v | tr ' =,' '_-+')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
