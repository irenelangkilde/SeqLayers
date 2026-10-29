#!/bin/zsh
# chainsel = z0: selective (positive-part James-Stein) shrinkage of chain steps toward their reference (the step with
# the feature's nearest generalization, else the previous step's odds), w = max(0, 1 - z0^2/z^2), z = deviation / SE.
# z0 = 2, 2.6 (~ expected max of ~30 noise z-scores: maxdev's selection), 3.5.  Bases: B (0.8531 / 0.8020) and DC
# (accum mindevsup + explainpool + depweight + prodprops, 0.8272 / 0.7779).  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/chainsel
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
DC="accumby=mindevsup explainpool=true depweight=true prodprops=head_xpos,head_dir,head_dist"
{
for base in B DC; do
  for z in 2 2.6 3.5; do
    v="chainsel=$z"
    [[ $base == DC ]] && v="$v $DC"
    n=${base}_z$z
    echo "=== $base z0 $z  $(date)"
    julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=chainsel_$n > $OUT/$n.log 2>&1
    grep -E '^ALL|^chainsel|^score|ERROR' $OUT/$n.log
  done
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
