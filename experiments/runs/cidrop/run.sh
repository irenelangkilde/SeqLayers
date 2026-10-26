#!/bin/zsh
# Accum leftovers with a conditional-independence drop rule (cidrop=θ): a context feature B may be
# dropped (backoff's last step) only if some feature A kept has D(A→B) < θ, D from the pairwise table
# (experiments/ci_table.jl, attachment decisions; other decision types drop as before).  Base: soft
# pool + greedyby=odds + leftsmooth=evidence + product domains eps 0 + ctxmode=accum (all decisions):
# 0.7937 / 0.7485.  Slice, Mac, 6 workers.  θ = 0.05, 0.1, 0.25.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
OUT=experiments/runs/cidrop
W=experiments/work/train500-CHILDES+ESLSpok+EWT+GUM+GUMReddit-gold-rootabs
julia --project=experiments -e "include(\"experiments/dev_eval.jl\"); write(\"$W/stamp\", srcstamp())"
COMMON=(model=chain select=maxdev events=gold dev_sents=5 jointmin=10000 leftovers=greedy minctx=2 relevance=0.25 gate=0.5
  countmin=10000 rootabsent=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true greedyby=odds
  leftsmooth=evidence domains=product domeps=0 ctxmode=accum)
{
echo "=== ci table  $(date)"
julia --project=experiments experiments/ci_table.jl "${COMMON[@]}" > $OUT/ci_table.log 2>&1
grep -v "^\[ Info" $OUT/ci_table.log | tail -8
for th in 0.05 0.1 0.25; do
  echo "=== cidrop $th  $(date)"
  julia --project=experiments experiments/ceiling.jl "${COMMON[@]}" workers=6 tags=false headconf=1.0 cidrop=$th note=accum-cidrop$th > $OUT/cidrop$th.log 2>&1
  grep -E '^ALL|^score|ERROR' $OUT/cidrop$th.log
done
echo "=== all done $(date)"
} > $OUT/summary.txt 2>&1
touch $OUT/done
