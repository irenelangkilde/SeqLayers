#!/bin/zsh
# explainpool=true: an explained-away leftover (mindev diff < 0.1) joins the context pool with correction 0 instead of
# being discarded, so later leftovers can condition on it.  Reruns of the discarding runs:
#   accum (base B):  mindev 0.8424 / 0.7944, mindevsup 0.8424 / 0.7927
#   greedy:  mindev exhaust greedysel leftmin2=5  0.8330 / 0.7789;  + depweight + prodprops  0.8272 / 0.7713;
#            mindev exhaust greedysel (no leftmin2) + depweight + prodprops  0.8255 / 0.7713
# Slice ceiling, Mac, 6 workers, dcstats.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/explainpool
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  leftsmooth=evidence projective=true domains=product domeps=0 headtrank=true dcstats=true explainpool=true)
L="depweight=true prodprops=head_xpos,head_dir,head_dist"
G="ctxmode=greedy greedyby=mindev mindevstop=exhaust greedysel=true"
{
for v in "ctxmode=accum greedyby=odds accumby=mindev" "ctxmode=accum greedyby=odds accumby=mindevsup" \
  "$G leftmin2=5" "$G leftmin2=5 $L" "$G $L"; do
  n=$(echo $v | sed -e 's/prodprops=[^ ]*/prodprops/' -e 's/greedyby=mindev mindevstop=exhaust greedysel=true/mindev/' -e 's/ greedyby=odds//' | tr ' ,' '_+' | tr '=' '-')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^double|^explainpool|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
