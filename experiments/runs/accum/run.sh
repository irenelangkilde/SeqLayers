#!/bin/zsh
# Leftovers as a second chain over generalized contexts (ctxmode=accum): each round, every remaining
# leftover is scored against the accumulated context (chain + leftovers taken so far, generalized only as
# far as it needs, keeping the context's odds closest to the running estimate); the largest |correction|
# is taken and joins the context.  1. for the attachment and head properties (ctxscope=heads), 2. for all
# decisions.  Base: soft pool + greedyby=odds + leftsmooth=evidence + product domains eps 0; compare
# backoff heads 0.7868 / 0.7420, greedy 0.7844 / 0.7406.  Slice, Mac, 6 workers.  Then the error traces.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/accum
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25 gate=0.5
  countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence domains=product domeps=0 ctxmode=accum)
run() { name=$1; shift; echo "=== $name  $(date)"; julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=6 tags=false headconf=1.0 "$@" note=$name > $OUT/$name.log 2>&1;
  grep -E '^ALL|^score|ERROR' $OUT/$name.log; }
{
run accum-heads ctxscope=heads
run accum-all
echo "=== explain  $(date)"
julia --project=experiments experiments/ceiling_explain.jl "${COMMON[@]}" ctxscope=heads docs=4 nsent=2 maxerr=2 maxlen=16 explaintop=2 > $OUT/explain.log 2>&1
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
