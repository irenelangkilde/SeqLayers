#!/bin/bash
# Forked chains (user 2026-11-02) on base B + kin + nn2 (0.8782 / 0.8230): where the largest |step| lowers the odds
# and another feature raises them most, the chain forks (forkcap=16).  forkcomb max | avg; forkleft each (leftovers per
# chain) | chosen (best chain by its odds gets leftovers); leftovers taken by largest |corr| (accumpick=abs) or largest
# signed corr (signed).  Batches: 1 signed leftovers alone (records events); 2 forks abs; 3 forks signed; 4 rerun of best
# + bioes (crashed in runs/oclass-bioes), alone.  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/fork
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true
  neighbor2=true kinitems=true headkin=left,right)
one() {
  n=fork_$(echo "$1" | tr ' ,' '_+' | tr '=' '-')   # (GNU tr reads '_-+' as a range)
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=$2 $1 note=$n > $OUT/$n.log 2>&1
  { echo "=== $1"; grep -E '^ALL|^forkcap|^score|recorded|ERROR' $OUT/$n.log; } >> $OUT/summary-$TAG.txt
}
echo "=== start $(date)" > $OUT/summary-$TAG.txt
one "accumpick=signed" 14
for pk in abs signed; do
  one "forkcap=16 forkcomb=max forkleft=each accumpick=$pk" 5 & sleep 20
  one "forkcap=16 forkcomb=max forkleft=chosen accumpick=$pk" 5 & sleep 20
  one "forkcap=16 forkcomb=avg forkleft=each accumpick=$pk" 5 & wait
done
one "idxeq=true bioes=true" 14
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
