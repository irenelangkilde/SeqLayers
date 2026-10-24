#!/bin/bash
# Profile the soft pool model (sigma 0.1, drop weak non-redundant features, keep generalizations;
# 0.7478 / 0.7044 on the slice, 941 s on the Mac with 6 workers): gate 0.5 + relevance 0.25,
# redundant pool, rootabsent; noise=20 events (noise is gated out of attachment).  One process:
# 1 warm-up doc, then the next 3 smallest dev docs profiled, memo emptied in between.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/profile-softpool
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
/usr/bin/time -v julia --project=experiments experiments/profile_chain.jl model=chain select=maxdev events=gold dev_sents=5 \
  jointmin=10000 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 redundant=true \
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true docs=3 warm=1 flat=200 > $OUT/profile.txt 2>&1
touch $OUT/done
