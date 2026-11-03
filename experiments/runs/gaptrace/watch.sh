#!/bin/bash
# Copy the gap trace back from First Worker when done (no shutdown: the fast decode's watcher does that).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
H=ubuntu@3.148.105.132
t0=$(date +%s)
until $K -n $H "f=$P/experiments/runs/gaptrace/done-gap; test -f \$f && [ \$(stat -c %Y \$f) -gt $t0 ]"; do sleep 60; done
rsync -az -e "$K" $H:$P/experiments/runs/gaptrace/gap-kinnn2.log $P/experiments/runs/gaptrace/
echo "copied back $(date)"
