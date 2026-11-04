#!/bin/bash
# Mac watcher (ip, tag): when the bipolar run is done, copy runs/bipolar back, append its csv rows here, and shut
# the instance down 5 minutes later (user's standing request).  Only a marker newer than this watcher counts.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
t0=$(date +%s)
until $K -n ubuntu@$1 "f=$P/experiments/runs/bipolar/done-$2; test -f \$f && [ \$(stat -c %Y \$f) -gt $t0 ]"; do sleep 60; done
rsync -az -e "$K" ubuntu@$1:$P/experiments/runs/bipolar/ $P/experiments/runs/bipolar/
for f in ceiling.csv ceiling_breakdown.csv results.csv; do
  $K -n ubuntu@$1 "grep '\"bipolar_' $P/experiments/$f" >> $P/experiments/$f
done
$K -n ubuntu@$1 "sudo shutdown -h +5" && echo "$2 ($1): copied back, shutdown scheduled in 5 min, $(date)"
cat $P/experiments/runs/bipolar/summary-$2.txt
