#!/bin/zsh
# greedyby=mindev (user's design 2026-10-28): for leftover e, B starts empty and grows by the a in A \ B (chain,
# redundant pool, taken leftovers) with the smallest diff = |log o(x|e,B∪a) - log o(x|B∪a)|; e is explained away
# (dropped) once diff < 0.1.  Stop (mindevstop): localmin (every a would raise diff), exhaust (nothing supported;
# last diff), pathmin (smallest diff on the path).  greedysel: the leftover with the largest |diff| is taken first
# and joins A.  Two bases, ctxmode=greedy: g = base B greedy (greedyby=odds 0.8399 / 0.7879 earlier);
# l = g + depweight + prodprops=head_xpos,head_dir,head_dist.  dcstats on.  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/mindev
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  leftsmooth=evidence ctxmode=greedy projective=true domains=product domeps=0 headtrank=true dcstats=true)
L="depweight=true prodprops=head_xpos,head_dir,head_dist"
M="greedyby=mindev greedysel=true mindevstop"
{
for v in "greedyby=odds" "greedyby=odds greedysel=true" "$M=localmin" "$M=exhaust" "$M=pathmin" \
  "greedyby=odds greedysel=true $L" "$M=localmin $L" "$M=exhaust $L" "$M=pathmin $L"; do
  n=$(echo $v | sed 's/prodprops=[^ ]*/prodprops/' | tr ' =,' '_-+')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^double|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
