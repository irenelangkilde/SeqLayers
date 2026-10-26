#!/bin/bash
# Relations + projective with the current best leftover setup: soft pool (sigma 0.1, drop + keepgen) +
# greedyby=odds + leftsmooth=evidence + ctxmode=accum, with product domains (eps 0) and without.  2 workers,
# one after the other (relations workers need 20-49 GB).  Alongside, the no-relations control on the same
# (noise-20) events, 4 workers.  Compare: relations + projective, Oct 23 base: 0.8706 / 0.8092; cheap best +
# projective + product (Mac): 0.8465 / 0.7927.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/relproj-accum
for W in experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs-rel; do
  julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
done
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy minctx=2
  relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|^pipeline\|Elapsed (wall\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{
  run relproj-accum-product0 relations=true workers=2 maxcached=50000000 domains=product domeps=0
  run relproj-accum relations=true workers=2 maxcached=50000000
} > $OUT/rel.txt 2>&1 &
{
  run proj-accum-product0-control workers=4 domains=product domeps=0
} > $OUT/control.txt 2>&1 &
wait
cat $OUT/rel.txt $OUT/control.txt > $OUT/summary.txt
touch $OUT/done
