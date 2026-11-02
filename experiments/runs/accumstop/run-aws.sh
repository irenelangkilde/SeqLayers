#!/bin/bash
# accum that stops (user 2026-11-02): once the best leftover's backed-off context drifts more than theta from T
# (|log o(C) - T| > theta), it and all remaining leftovers are discarded (accumstop=theta).  Base B (0.8531 / 0.8020);
# switching to greedy contexts instead (accumgreedy): 0.5 0.8482, 1 0.8510, 2 0.8479.
# Args: tag, then one quoted option string per run; side by side, 5 workers each; marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/accumstop
TAG=$1; shift
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs-htrank
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true
)
one() {
  n=astop_$(echo "$1" | tr ' ,' '_+' | tr '=' '-')   # (GNU tr reads '_-+' as a range)
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" $1 note=$n > $OUT/$n.log 2>&1
  { echo "=== $1"; grep -E '^ALL|^interleave|^posnegrestart|^score|ERROR' $OUT/$n.log; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
first=1
for v in "$@"; do
  [ $first = 1 ] || sleep 30
  first=0
  one "$v" &
done
wait
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
