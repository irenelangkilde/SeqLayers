#!/bin/bash
# Kinship + neighbor2 (kinitems, headkin=left,right, neighbor2) on the three open fronts (user 2026-11-02), one after the
# other (the relations workers need 20-50 GB each; runs 1 and 3 share an events dir):
#   1 posneg-heads ceiling (0.8578 / 0.8075; kin + nn2 on base B 0.8782 / 0.8230), 12 workers
#   2 relations ceiling, the 0.8789 / 0.8196 command (noise 20, relations, prodprops=head_xpos,head_dir), 2 workers
#   3 full decode, the best decode (softheads + head_trank + posneg heads, 0.6936 / 0.6280), 14 workers
# Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/best3
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
KN="neighbor2=true kinitems=true headkin=left,right"
PN="stepscore=posneg posnegrestart=true posnegafter=true posnegonly=heads"
echo "=== start $(date)" > $OUT/summary-$TAG.txt
# 1 posneg-heads ceiling
n=best3_posnegheads_kinnn2
echo "=== 1 $n  $(date)" >> $OUT/summary-$TAG.txt
julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=12 tags=false \
  jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true \
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true \
  domains=product domeps=0 headtrank=true dcstats=true $PN $KN note=$n > $OUT/$n.log 2>&1
grep -E '^ALL|^posnegrestart|^head_kin|^score|recorded|ERROR' $OUT/$n.log >> $OUT/summary-$TAG.txt
# 2 relations ceiling
n=best3_relations_kinnn2
echo "=== 2 $n  $(date)" >> $OUT/summary-$TAG.txt
julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 \
  headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 redundant=true \
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true relations=true \
  workers=2 maxcached=50000000 domains=product domeps=0 dcstats=true prodprops=head_xpos,head_dir $KN note=$n > $OUT/$n.log 2>&1
grep -E '^ALL|^head_kin|^score|recorded|ERROR' $OUT/$n.log >> $OUT/summary-$TAG.txt
# 3 full decode
n=best3_decode_posnegheads_kinnn2
echo "=== 3 $n  $(date)" >> $OUT/summary-$TAG.txt
julia --project=experiments experiments/dev_eval.jl model=chain select=maxdev events=gold dev_sents=5 workers=14 \
  jointmin=10000 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
  poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true \
  $PN $KN note=$n > $OUT/$n.log 2>&1
grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^build|ERROR|recorded events" $OUT/$n.log >> $OUT/summary-$TAG.txt
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
