#!/bin/bash
# Mac watcher (ip, tag) for the context-cache run: copies back only the ctxcache_(off|on) row (the
# workers keep their disks, so their csvs still hold earlier decode2_ rows).
# It then shuts the instance down 5 minutes later (user's standing request).  Only a marker newer than this watcher counts.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
t0=$(date +%s)
until $K -n ubuntu@$1 "f=$P/experiments/runs/ctxcache/done-$2; test -f \$f && [ \$(stat -c %Y \$f) -gt $t0 ]"; do sleep 60; done
rsync -az -e "$K" ubuntu@$1:$P/experiments/runs/ctxcache/ $P/experiments/runs/ctxcache/
for f in results.csv; do
  $K -n ubuntu@$1 "grep -E '\"ctxcache_(off|on)\"' $P/experiments/$f" >> $P/experiments/$f
done
$K -n ubuntu@$1 "sudo shutdown -h +5" && echo "$2 ($1): copied back, shutdown scheduled in 5 min, $(date)"
cat $P/experiments/runs/ctxcache/summary-$2.txt
