#!/bin/zsh
# mindev with two-sided support (leftmin2=k: a competes only if B∪a∪e has >= k chosen and >= k not-chosen records).
# Traces (runs/ceiling-explain/explain-mindev-counts.log): most "explained away" diffs ~0 came from contexts with
# 0 chosen records with e (0/5, 0/193, 0/759) -- no evidence, not redundancy.  Without it: exhaust 0.8368 / 0.7820
# (g), 0.8255 / 0.7713 (l = + depweight + prodprops).  Controls: odds + greedysel with leftmin2=5.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/mindev
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  leftsmooth=evidence ctxmode=greedy projective=true domains=product domeps=0 headtrank=true dcstats=true greedysel=true)
L="depweight=true prodprops=head_xpos,head_dir,head_dist"
M="greedyby=mindev mindevstop=exhaust"
{
for v in "$M leftmin2=3" "$M leftmin2=5" "$M leftmin2=10" "greedyby=odds leftmin2=5" \
  "$M leftmin2=3 $L" "$M leftmin2=5 $L" "$M leftmin2=10 $L" "greedyby=odds leftmin2=5 $L"; do
  n=min2_$(echo $v | sed -e 's/prodprops=[^ ]*/prodprops/' -e 's/greedyby=mindev mindevstop=exhaust/mindev/' | tr ' =,' '_-+')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^double|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary-min2.txt 2>&1
touch $OUT/done-min2
