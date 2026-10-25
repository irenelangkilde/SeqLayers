#!/bin/zsh
# Backoff leftover contexts for the attachment decision only (ctxattach): labels and tags (deprel, head
# properties, ...) keep greedy contexts.  Base: soft pool + greedyby=odds + leftsmooth=evidence + product
# domains eps 0 (0.7844 / 0.7406); backoff for all decisions 0.7868 / 0.7285.  Slice, Mac, 6 workers.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/ctxattach
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
julia --project=experiments experiments/ceiling.jl model=chain select=maxdev events=gold dev_sents=5 workers=6 tags=false \
  jointmin=10000 headconf=1.0 leftovers=greedy minctx=2 relevance=0.25 gate=0.5 countmin=10000 rootabsent=true redundant=true \
  poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds leftsmooth=evidence domains=product domeps=0 \
  ctxmode=backoff ctxattach=true note=backoff-attach > $OUT/backoff-attach.log 2>&1
grep -E "^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^score|ERROR" $OUT/backoff-attach.log > $OUT/summary.txt
touch $OUT/done
