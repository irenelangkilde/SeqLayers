#!/bin/bash
# Mac watcher: per machine (ip tag), when its gold-token decode are done, copy runs/goldtok back, append its
# ceiling.csv rows here, and shut it down 5 minutes later (user's standing request).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
watch1() {
  # only a marker newer than this watcher counts (instances are stopped, not terminated, so old markers survive)
  local t0=$(date +%s)
  until $K -n ubuntu@$1 "f=$P/experiments/runs/goldtok/done-$2; test -f \$f && [ \$(stat -c %Y \$f) -gt $t0 ]"; do sleep 60; done
  rsync -az -e "$K" ubuntu@$1:$P/experiments/runs/goldtok/ $P/experiments/runs/goldtok/
  for f in ceiling.csv ceiling_breakdown.csv results.csv; do
    $K -n ubuntu@$1 "grep '\"goldtok_' $P/experiments/$f" >> $P/experiments/$f
  done
  $K -n ubuntu@$1 "sudo shutdown -h +5" && echo "$2 ($1): copied back, shutdown scheduled in 5 min, $(date)"
}
watch1 3.139.234.189 awsB &
wait
cat $P/experiments/runs/goldtok/summary-awsB.txt
