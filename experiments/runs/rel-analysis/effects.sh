#!/bin/bash
# rel_effects.jl alone (run.sh passed it headconf, a ceiling-only option); alongside run.sh's usage pass.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/rel-analysis
/usr/bin/time -v julia --project=experiments experiments/rel_effects.jl model=chain select=maxdev events=gold dev_sents=5 jointmin=10000 \
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 relations=true > $OUT/effects.txt 2>&1
touch $OUT/effects_done
