#!/bin/zsh
# Index-equivalence attachment items (idxeq=true; user's idea 2026-10-31): same-token tests between the token's
# neighbours and the candidate's nearest dependents / head.  New events (-htrank-idxeq).  On base B (0.8531 / 0.8020)
# and the posneg heads setup (0.8579 / 0.8075).  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/idxeq
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true idxeq=true)
{
for v in "stepscore=abs" "stepscore=posneg posnegrestart=true posnegafter=true posnegonly=heads"; do
  n=idxeq_$( [[ $v == stepscore=abs ]] && echo base || echo posnegheads )
  echo "=== idxeq $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^score|recorded|ERROR' $OUT/$n.log
done
} > $OUT/summary.txt 2>&1
