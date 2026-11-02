#!/bin/bash
# Mac watcher: per machine (ip tag), when its association runs are done, copy runs/assoc back, append its
# ceiling.csv rows here, and shut it down 5 minutes later (user's standing request).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
watch1() {
  # only a marker newer than this watcher counts (instances are stopped, not terminated, so old markers survive)
  local t0=$(date +%s)
  until $K -n ubuntu@$1 "f=$P/experiments/runs/assoc/done-$2; test -f \$f && [ \$(stat -c %Y \$f) -gt $t0 ]"; do sleep 60; done
  rsync -az -e "$K" ubuntu@$1:$P/experiments/runs/assoc/ $P/experiments/runs/assoc/
  for f in ceiling.csv ceiling_breakdown.csv; do
    $K -n ubuntu@$1 "grep '\"assoc1_' $P/experiments/$f" >> $P/experiments/$f
  done
  $K -n ubuntu@$1 "sudo shutdown -h +5" && echo "$2 ($1): copied back, shutdown scheduled in 5 min, $(date)"
}
watch1 13.184.184.113 awsA &
watch1 3.136.19.179 awsB &
wait
cat $P/experiments/runs/assoc/summary-awsA.txt $P/experiments/runs/assoc/summary-awsB.txt
