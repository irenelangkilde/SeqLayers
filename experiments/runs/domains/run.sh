#!/bin/bash
# Domain variables in the attachment pipeline (ceiling.jl domains=): the attachment and the head
# properties keep domains (values with probability >= domeps), narrowed together by propagation of
# head_X(t) = X(attach(t)); the attachment chosen over its domain by its own probability (prune) or
# times the head properties' probabilities of its values (product).  Model: soft pool sigma 0.1 + drop
# weak + keep generalizations, gate 0.5 + relevance 0.25, redundant, rootabsent; noise=20 events
# (gated out of attachment); slice; 6 workers.  Baseline (headconf 1.0, nothing committed) on the
# same events first.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
export PATH=$HOME/.juliaup/bin:$PATH
OUT=experiments/runs/domains
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-noise20-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false jointmin=10000 headconf=1.0
  leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true noise=20 redundant=true
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true)
run() { name=$1; shift; echo "=== $name  $(date)"; /usr/bin/time -v julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" "$@" note=$name > $OUT/$name.log 2>&1;
  grep -A1 "^ALL\|^score\|ERROR\|^pipeline\|^domains\|Elapsed (wall\|using cached\|Maximum resident" $OUT/$name.log | grep -v "^--"; }
{
run baseline
run prune-1e-3 domains=prune domeps=1e-3
run product-1e-3 domains=product domeps=1e-3
run product-1e-2 domains=product domeps=1e-2
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
