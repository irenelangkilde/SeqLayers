#!/bin/bash
# Mac watcher (ip, tag) for the relstrikes run: as watch.sh, but copies back only the relstrikes_ rows (the
# workers keep their disks, so their csvs still hold the earlier thin_ rows).
# It then shuts the instance down 5 minutes later (user's standing request).  Only a marker newer than this watcher counts.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
t0=$(date +%s)
until $K -n ubuntu@$1 "f=$P/experiments/runs/relstrikes/done-$2; test -f \$f && [ \$(stat -c %Y \$f) -gt $t0 ]"; do sleep 60; done
rsync -az -e "$K" ubuntu@$1:$P/experiments/runs/relstrikes/ $P/experiments/runs/relstrikes/
for f in ceiling.csv ceiling_breakdown.csv results.csv; do
  $K -n ubuntu@$1 "grep -E '\"relstrikes_' $P/experiments/$f" >> $P/experiments/$f
done
$K -n ubuntu@$1 "sudo shutdown -h +5" && echo "$2 ($1): copied back, shutdown scheduled in 5 min, $(date)"
cat $P/experiments/runs/relstrikes/summary-$2.txt
