#!/bin/bash
# Double-counting reductions and head-term changes on the relations ceiling (relations + projective + accum + product,
# noise-20 relation events: 0.8769 / 0.8179; no punct 0.8996 / 0.8289).  Traces: leftovers stack correlated relation
# features (a dependent's upos and deprel), and head_dist / head_upos terms flip right attachments.
# One at a time, 2 workers (relation workers need 20-50 GB).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/rel-lowdc
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs-rel
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2
  relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true relations=true workers=2 maxcached=50000000
  domains=product domeps=0 dcstats=true)
{
for v in "depweight=true" "telescope=true accumreset=true" "telescope=true accumreset=true depweight=true" \
  "accumby=mindevsup explainpool=true" "prodprops=head_upos,head_xpos,head_dir" "prodprops=head_xpos,head_dir"; do
  n=rel_$(echo $v | tr ' ,' '_+' | tr '=' '-')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" $v note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^double|^explainpool|^score|ERROR' $OUT/$n.log
  grep -h "$n" experiments/ceiling_breakdown.csv > $OUT/$n.breakdown.csv
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
