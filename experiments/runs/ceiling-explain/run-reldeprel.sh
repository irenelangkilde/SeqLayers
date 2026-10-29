#!/bin/bash
# Traces of errors by gold deprel (conj, parataxis, acl, advcl) on the relations ceiling (relations + projective +
# accum + product domains, noise-20 relation events: 0.8769 / 0.8179).  Any error of that deprel (goldfix=false),
# up to 2 sentences each, detailed chains and accum leftovers.  Two at a time (relation workers need 20-50 GB).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/ceiling-explain
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs-rel
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=1 jointmin=10000 headconf=1.0 leftovers=greedy minctx=2
  relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true
  greedyby=odds leftsmooth=evidence ctxmode=accum projective=true relations=true maxcached=50000000 domains=product domeps=0
  goldfix=false docs=40 nsent=2 maxerr=3 maxlen=25 explaintop=2 detail=true)
tr1() { julia --project=experiments experiments/ceiling_explain.jl "${COMMON[@]}" deprel=$1 > $OUT/detail-rel-$1.log 2>&1; }
tr1 conj & tr1 parataxis & wait
tr1 acl & tr1 advcl & wait
touch $OUT/done-reldeprel
