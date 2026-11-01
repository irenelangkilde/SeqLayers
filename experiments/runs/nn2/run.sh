#!/bin/zsh
# neighbor2 (user's idea 2026-11-01): left_neighbor^left_neighbor and right_neighbor^right_neighbor .{form, upos, head_dir}
# as context and attachment paths (both sides); on base B (0.8531 / 0.8020) and base B + idxeq (0.8562 / 0.8051).
# New specs: each records its own events.  Mac, 6 workers, one at a time.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/nn2
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true)
{
for v in "neighbor2=true" "idxeq=true neighbor2=true"; do
  n=nn2_$(echo $v | tr ' ' '_' | tr '=' '-')
  echo "=== $v  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" ${=v} note=$n > $OUT/$n.log 2>&1
  grep -E '^ALL|^score|ERROR' $OUT/$n.log
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
