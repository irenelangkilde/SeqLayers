#!/bin/bash
# Contest re-ranking without leftovers anywhere (so the contest scores and the third candidate's
# share a scale): base gate 0.5, rootabsent, noise=20, leftovers=none; slice, warm.  Compare with
# noleft-indep (UAS 0.7358 / LAS 0.6864).  (src changed after these events were recorded: chain.jl
# and harness options only; stamp refreshed.)
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/contest-noleft
/usr/bin/time -v julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 \
  tags=false jointmin=10000 headconf=1.0 leftovers=none gate=0.5 countmin=10000 rootabsent=true noise=20 \
  contest=true maxrestarts=2 note=contest-noleft > $OUT/run.log 2>&1
touch $OUT/done
