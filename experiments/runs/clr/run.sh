#!/bin/bash
# Shared-chain selection (select=clr_rms: Aitchison distance between the candidates' log-odds vectors before
# and after each step; one feature sequence for all candidates, per-candidate gates and leftovers) vs maxdev
# (a chain per candidate), on the cheap best setup: soft pool + greedyby=odds + leftsmooth=evidence +
# ctxmode=accum + product domains (eps 0) + projective (maxdev on the Mac: 0.8465 / 0.7927).
# Slice ceiling, both at once, 6 workers each.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/clr
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2
  relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{ run clr_rms select=clr_rms; } > $OUT/clr.txt 2>&1 &
{ run maxdev select=maxdev; } > $OUT/maxdev.txt 2>&1 &
wait
cat $OUT/clr.txt $OUT/maxdev.txt > $OUT/summary.txt
touch $OUT/done
