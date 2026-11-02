#!/bin/zsh
# Association step scores with one-count smoothing (user 2026-11-02: counts are discrete, so less than one count is not
# enough): among the steps moving the odds by tau, the feature most associated with the candidate's choice within the
# prefix -- stepscore=pmi|mi|oddspmi|moi|logor (see step_crit).  Base B (0.8531 / 0.8020).  The half-count runs are in
# half-count/.  Mac, two at a time, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/assoc
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
one() {
  n=assoc1_stepscore-$1
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" stepscore=$1 note=$n > $OUT/$n.log 2>&1
  { echo "=== stepscore=$1"; grep -E '^ALL|^score|ERROR' $OUT/$n.log; } >> $OUT/summary.txt
}
echo "=== start $(date)" > $OUT/summary.txt
one pmi & sleep 30; one mi & wait
one oddspmi & sleep 30; one moi & wait
one logor
echo "=== all done $(date)" >> $OUT/summary.txt
touch $OUT/done
