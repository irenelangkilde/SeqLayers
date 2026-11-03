#!/bin/bash
# Reruns of the two chain-first forked runs (forkleft=chosen), killed by the OOM killer when three forked runs shared the
# machine (forked workers grow to 10-14 GB): one at a time, 6 workers, maxcached 50M.  Waits for the first batch
# (done-awsD); marks done-awsD2.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/fork
rm -f "$OUT/done-awsD2"
until [ -f $OUT/done-awsD ]; do sleep 60; done
COMMON=(model=chain select=maxdev events=gold dev_sents=5 tags=false jointmin=10000 headconf=1.0 leftovers=greedy
  minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 headtrank=true dcstats=true
  neighbor2=true kinitems=true headkin=left,right workers=6 maxcached=50000000)
for pk in abs signed; do
  v="forkcap=16 forkcomb=max forkleft=chosen accumpick=$pk"
  n=fork_$(echo "$v" | tr ' ,' '_+' | tr '=' '-')
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" $v note=$n > $OUT/$n.log 2>&1
  { echo "=== (rerun) $v"; grep -E '^ALL|^forkcap|^score|ERROR' $OUT/$n.log; } >> $OUT/summary-awsD.txt
done
echo "=== reruns done $(date)" >> $OUT/summary-awsD.txt
touch $OUT/done-awsD2
