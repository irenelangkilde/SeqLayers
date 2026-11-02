#!/bin/zsh
# Fifth association step score (user 2026-11-02): |log odds ratio| (stepscore=logor), after the first four finish.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/assoc
until [ -f $OUT/done ]; do sleep 30; done
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=8 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
n=assoc_stepscore-logor
julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" stepscore=logor note=$n > $OUT/$n.log 2>&1
{ echo "=== stepscore=logor"; grep -E '^ALL|^score|ERROR' $OUT/$n.log; echo "=== logor done $(date)"; } >> $OUT/summary.txt
touch $OUT/done-logor
