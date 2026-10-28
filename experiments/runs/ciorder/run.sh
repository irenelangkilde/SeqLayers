#!/bin/zsh
# Accum walk ordered by conditional dependence (ciorder): each generalization step moves the context feature B
# that depends least on the rest -- slot: mean of D[A, B] over all slots A (one fixed order); value: mean over the
# context's other features A = a of D(a→B) (value-level table, values with >= 200 records in the 200k sample;
# D[A, B] for rarer a).  Attachment decisions only (other types keep the odds-closest step).  Base: best ceiling
# setup (head_trank) 0.8531 / 0.8020; control rerun at the end.  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/ciorder
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25 gate=0.5
  countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true)
{
echo "=== ci tables  $(date)"
julia --project=experiments experiments/ci_table.jl "${COMMON[@]}" > $OUT/ci_table.log 2>&1
grep -v "^\[ Info" $OUT/ci_table.log | tail -10
for o in slot value off; do
  echo "=== ciorder $o  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=6 tags=false headconf=1.0 ciorder=$o note=ciorder-$o > $OUT/ciorder-$o.log 2>&1
  grep -E '^ALL|^score|ERROR' $OUT/ciorder-$o.log
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
