#!/bin/zsh
# Tree rank as an attachment item (treerank): for each candidate head, t's rank among its dependents on t's
# side counting outward (1 + the candidate's dependents between it and t; buckets 1, 2, 3, 4+, root).  Current
# best setup: soft pool + greedyby=odds + leftsmooth=evidence + accum + projective + product domains (eps 0):
# 0.8465 / 0.7927 without.  Slice ceiling, Mac, 6 workers; records new events (-trank).  Then the four traced
# sentences again.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/treerank
COMMON=(model=chain select=maxdev events=gold dev_sents=5 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25 gate=0.5
  countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence ctxmode=accum projective=true domains=product domeps=0 treerank=true)
{
echo "=== treerank  $(date)"
julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=6 tags=false headconf=1.0 note=treerank > $OUT/treerank.log 2>&1
grep -E '^(CHILDES|ESLSpok|EWT|GUM|GUMReddit|ALL) |^score|ERROR|recorded events' $OUT/treerank.log
echo "=== explain  $(date)"
julia --project=experiments experiments/ceiling_explain.jl "${COMMON[@]}" docs=10 nsent=4 maxerr=2 maxlen=16 explaintop=2 > $OUT/explain.log 2>&1
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
