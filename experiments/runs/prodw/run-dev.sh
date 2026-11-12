#!/bin/bash
# Learned product weights (user 2026-11-12), step 2: on dev, the weights fitted on the first 9.7k held-out tune decisions
# (fit_prodw.jl, lambda 1; scaled so attach = 1 -- only the argmax matters), then the plain control with the dev dump
# (the Mac control timing; offline check of the weights).  Mac, one at a time.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/prodw
rm -f $OUT/done-dev
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true
  neighbor2=true kinitems=true headkin=left,right relstrikes=2)
W=attach=1,head_dir=0.891,head_dist=0.701,head_kinleft=1.506,head_kinright=0.81,head_trank=0.333,head_upos=0.874,head_xpos=0.144
echo "=== start $(date)" > $OUT/summary-dev.txt
julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=6 prodw=$W note=prodw_tune1 > $OUT/prodw_tune1.log 2>&1
{ echo "=== prodw $W"; grep -E '^ALL|^score|recorded|ERROR' $OUT/prodw_tune1.log; } >> $OUT/summary-dev.txt
julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=6 proddump=$OUT/terms-dev note=control_mac > $OUT/control_mac.log 2>&1
{ echo "=== control (plain product, dev dump)"; grep -E '^ALL|^score|recorded|ERROR' $OUT/control_mac.log; } >> $OUT/summary-dev.txt
echo "=== all done $(date)" >> $OUT/summary-dev.txt
touch $OUT/done-dev
