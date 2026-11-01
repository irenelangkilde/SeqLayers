#!/bin/bash
# Product without head_dist / head_upos on the no-relations bests (the change that gave the best relations ceiling,
# 0.8789 / 0.8196): prodprops=head_xpos,head_dir and head_xpos,head_dir,head_trank on posneg heads (0.8579 / 0.8075)
# and on base B + idxeq (0.8562 / 0.8051).  12 workers.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/prodprops2
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=12 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true)
PN="stepscore=posneg posnegrestart=true posnegafter=true posnegonly=heads"
{
for base in posneg idxeq; do
  for pp in head_xpos,head_dir head_xpos,head_dir,head_trank; do
    if [ $base = posneg ]; then extra="$PN"; else extra="idxeq=true"; fi
    n=pp_${base}_$(echo $pp | tr ',' '+')
    echo "=== $base prodprops=$pp  $(date)"
    julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" $extra prodprops=$pp note=$n > $OUT/$n.log 2>&1
    grep -E '^ALL|^score|recorded|ERROR' $OUT/$n.log
    grep -h "$n" experiments/ceiling_breakdown.csv > $OUT/$n.breakdown.csv
  done
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
