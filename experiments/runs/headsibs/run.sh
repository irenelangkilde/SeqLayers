#!/bin/zsh
# Sibling head properties (soft, ceiling; user's request 2026-10-30): head_sibin / sibout / sibleft / sibright -- under
# its head, the token's nearest inner / outer / left / right sibling (none, distance bucket, root), predicted like the
# head properties and entering the product (domains=product) with the value each candidate head implies.
# All four recorded once (new events: -htrank-hsibinoutleftright); prodprops selects the product terms.
# Base B config (0.8531 / 0.8020).  Slice ceiling, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/headsibs
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true headsibs=in,out,left,right)
B=head_upos,head_xpos,head_dir,head_dist,head_trank
{
for v in "prodprops=$B" "prodprops=$B,head_sibin,head_sibout" "prodprops=$B,head_sibleft,head_sibright" ""; do
  n=sib_$( [[ -z $v ]] && echo all || echo $v | sed -e "s/$B/base/" -e 's/prodprops=//' -e 's/head_//g' | tr ',' '+')
  echo "=== ${v:-all four}  $(date)"
  /usr/bin/time julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^score|recorded|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
