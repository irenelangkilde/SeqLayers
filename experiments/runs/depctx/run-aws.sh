#!/bin/bash
# Leftover contexts chosen by the association measures (user 2026-11-02): depmeasure=mi|oddspmi|moi|logor (one-count
# smoothed, label-blind, on the leftover x context-feature table) in greedy contexts (ctxmode=greedy greedyby=dep;
# control greedy odds 0.8399 / 0.7879, dep with Yule 0.8364, pmi 0.8395) and accum backoff (accumby=depodds; control
# depodds with Yule 0.8513 / 0.8002, base B 0.8531 / 0.8020).  AWS, two batches of four, 4 workers each; marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/depctx
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=4 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
one() {
  n=depctx_$(echo "$1" | tr ' ,' '_+' | tr '=' '-')   # (GNU tr reads '_-+' as a range)
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" $1 note=$n > $OUT/$n.log 2>&1
  { echo "=== $1"; grep -E '^ALL|^score|ERROR' $OUT/$n.log; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
for M in mi oddspmi moi logor; do one "ctxmode=greedy greedyby=dep depmeasure=$M" & sleep 20; done; wait
for M in mi oddspmi moi logor; do one "accumby=depodds depmeasure=$M" & sleep 20; done; wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
