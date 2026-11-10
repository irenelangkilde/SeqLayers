#!/bin/bash
# Ceiling gains in the decode (user 2026-11-07): the fast kin + nn2 decode (candmemo + attachwait + rescoreevery 16,
# softheads, head_trank, posneg heads; 0.6947 / 0.6353) with stepmin=400 (ceiling 0.8831 / 0.8368 vs 0.8782 / 0.8230) or
# suffixes=3 (ceiling xpos 0.9676 -> 0.9717).  One config per machine (a kin + nn2 decode needs ~70 GB), 5 workers.
# Args: tag, name, options (comma-separated; '+' for a comma inside a value).  Marks done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/decode2
TAG=$1
rm -f "$OUT/done-$TAG"                 # (instances are stopped, not terminated: a marker from an earlier run may remain)
COMMON=(model=chain select=maxdev events=gold dev_sents=5 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25
  gate=0.5 countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence ctxmode=accum projective=true softheads=true headtrank=true stepscore=posneg posnegrestart=true
  posnegafter=true posnegonly=heads profdecode=true candmemo=true attachwait=true rescoreevery=16
  neighbor2=true kinitems=true headkin=left,right)
n=decode2_$2
echo "=== start $(date)" > $OUT/summary-$TAG.txt
julia --project=experiments experiments/dev_eval.jl "${COMMON[@]}" workers=5 $(echo "$3" | tr ',' ' ' | tr '+' ',') note=$n > $OUT/$n.log 2>&1
{ echo "=== $2"; grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^build|ERROR|recorded events| chars: |profdecode: token.attach" $OUT/$n.log | sed 's/^ *From worker [0-9]*:\t*//'; } >> $OUT/summary-$TAG.txt
echo "=== all done $(date)" >> $OUT/summary-$TAG.txt
touch $OUT/done-$TAG
