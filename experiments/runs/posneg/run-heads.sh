#!/bin/zsh
# posneg + restart after leftovers for attachment and head properties only (posnegonly=heads); abs for deprel, tags, gaps
#   attach   posneg + restart after leftovers for the attachment only; labeled decisions abs chains, no restart
#   negpos   negatives first, then positives (the mirror of posneg), no restart
#   reorder / reorderabs   restart after leftovers, rebuilt by appending the candidate's chain steps and taken
#            leftovers in the order of what each added before (signed / by size), re-scored
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/posneg
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
R="stepscore=posneg posnegrestart=true posnegafter=true"
{
for v in "$R posnegonly=heads"; do
  n=three_$(echo $v | sed -e 's/stepscore=posneg posnegrestart=true posnegafter=true/restartafter/' | tr ' ' '_' | tr '=' '-')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 \
    headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 \
    poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 \
    headtrank=true dcstats=true ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^posnegrestart|^score|ERROR' $OUT/$n.log
done
} > $OUT/summary-heads.txt 2>&1
