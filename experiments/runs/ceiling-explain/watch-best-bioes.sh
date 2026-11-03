#!/bin/bash
# Copy the best-bioes trace back from First Worker when done (no shutdown: the fast decode's watcher does that).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
H=ubuntu@3.148.105.132
t0=$(date +%s)
until $K -n $H "f=$P/experiments/runs/ceiling-explain/done-best-bioes; test -f \$f && [ \$(stat -c %Y \$f) -gt $t0 ]"; do sleep 60; done
rsync -az -e "$K" $H:$P/experiments/runs/ceiling-explain/detail-best-bioes.log $P/experiments/runs/ceiling-explain/
echo "copied back $(date)"; grep -c "" $P/experiments/runs/ceiling-explain/detail-best-bioes.log
