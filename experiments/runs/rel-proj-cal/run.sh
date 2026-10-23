#!/bin/bash
# 1. relations + projective with 3 workers (6 ran out of memory), warm from proj-rel/relations.
# 2. Calibration of the weight of the leftovers' corrections per decision type, on the tuning
#    sentences (base config, no relations), then the slice with that alpha applied.  Base: gate 0.5 +
#    relevance 0.25, rootabsent, noise=20.  Compare: relations 0.8048 / 0.7527; base 0.7399 / 0.6992.
# (src changed after these events were recorded, in options that leave them unchanged: stamps refreshed.)
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/rel-proj-cal
for W in experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs-rel; do
  julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
done
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|using cached\|Maximum resident\|alpha table" $OUT/$name.log | grep -v "^--"; }
{
run relations-projective workers=3 relations=true projective=true
run calibrate workers=6 on=tune calibrate=true
sed -n "/^calibration:/,/^alpha table/p" $OUT/calibrate.log
A=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs/events/alpha.tsv
cp $A $OUT/alpha.tsv
run leftalpha workers=6 leftalpha=$A
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
