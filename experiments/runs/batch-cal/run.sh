#!/bin/bash
# 1. Calibration of the weight of the leftovers' corrections per decision type on the tuning
#    sentences (3 workers, smaller chain cache), 2. the slice with that alpha, 3. pair steps (synergy)
#    with train_sents=2000 (2 workers).  Base: gate 0.5 + relevance 0.25 (+ redundant pool), rootabsent.
#    Compare: base with pool 0.7413 / 0.7009 (noise=20), pairs-synergy 0.7703 / 0.7296, train2000 0.7641 / 0.7168.
# (src changed after these events were recorded, with byte-identical events: stamps refreshed.)
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/batch-cal
for W in experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs experiments/work/train2000-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs; do
  julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
done
BASE=(model=chain select=maxdev events=gold dev_sents=5 maxcached=50000000 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${BASE[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|Elapsed (wall\|using cached\|Maximum resident\|alpha table" $OUT/$name.log | grep -v "^--"; }
{
run calibrate workers=3 noise=20 on=tune calibrate=true
sed -n "/^calibration:/,/^alpha table/p" $OUT/calibrate.log
A=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs/events/alpha.tsv
if [ -f $A ]; then cp $A $OUT/alpha.tsv; run leftalpha workers=6 noise=20 leftalpha=$A; else echo "no alpha table: skipping leftalpha"; fi
run train2000-pairs workers=2 train_sents=2000 pairsteps=true pairscore=synergy
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
