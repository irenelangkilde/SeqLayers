#!/bin/zsh
# Product domains: are the head-property terms overconfident / double counted?  Traces (runs/ceiling-explain/
# headblame.log): head_xpos 0.006 on VBG roots of sentence-final '.', head_dist 1 0.991 and head_trank 1 0.996 together
# outweighing a 0.9965 attachment ("a math class").  prodbeta: each term ^ beta; prodprops: only these properties in the
# product (the others still narrow domains): head_xpos subsumes head_upos, head_dist overlaps head_trank.
# Base: best ceiling setup (head_trank) 0.8531 / 0.8020.  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/prodterms
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true)
{
for v in "prodbeta=0.25" "prodbeta=0.5" "prodbeta=0.75" \
  "prodprops=head_xpos,head_dir,head_dist,head_trank" "prodprops=head_upos,head_xpos,head_dir,head_dist" \
  "prodprops=head_xpos,head_dir,head_dist" "prodprops=head_xpos,head_dir,head_dist prodbeta=0.5"; do
  n=$(echo $v | tr ' =,' '_-+')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
