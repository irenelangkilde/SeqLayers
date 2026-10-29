#!/bin/zsh
# New accum backoff-step rules on base B (accum + projective + product + head_trank, accumby=odds 0.8531 / 0.8020;
# dcstats smoke: 69.7% |Q| >= 0.5).  Each leftover X walks its own path; everything else as accum.
#   marg       the move keeping log o(x|C') closest to log o(x|X)
#   mindev     among moves whose context supports X, the smallest |log o(x|X,C') - log o(x|C')| (X explained away
#              below 0.1); until one supports X, the running-estimate rule
#   mindevsup  as mindev, but until one supports X, the move keeping the most records with X
#   mindev + leftmin2=5   supported = >= 5 chosen and >= 5 not-chosen records with X
# Waits for runs/mindev min2 to finish.  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
until [ -f experiments/runs/mindev/done-min2 ]; do sleep 60; done
OUT=experiments/runs/accumby2
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
{
for v in "accumby=marg" "accumby=mindev" "accumby=mindevsup" "accumby=mindev leftmin2=5"; do
  n=$(echo $v | tr ' =,' '_-+')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^double|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
