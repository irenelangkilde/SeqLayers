#!/bin/zsh
# chainshrink (user's request 2026-10-29): the chain's log odds from Beta-shrunk step sizes, the chain's features and
# order unchanged.  Targets: prev (previous step's odds), gen (the step with the feature's nearest generalization),
# prior (the final prefix's records toward the prior).  k = 20, 50, 200.  Bases:
#   B  = accum odds (0.8531 / 0.8020)
#   DC = accum mindevsup + explainpool + depweight + prodprops (0.8272 / 0.7779)
# Slice ceiling, Mac, 6 workers, dcstats.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/chainshrink
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
DC="accumby=mindevsup explainpool=true depweight=true prodprops=head_xpos,head_dir,head_dist"
{
for base in B DC; do
  for to in prev gen prior; do
    for k in 20 50 200; do
      v="chainshrink=$k chainshrinkto=$to"
      [[ $base == DC ]] && v="$v $DC"
      n=${base}_${to}_$k
      echo "=== $base $to k$k  $(date)"
      julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=chainshrink_$n > $OUT/$n.log 2>&1
      grep -E '^ALL|^double|^score|ERROR' $OUT/$n.log
    done
  done
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
