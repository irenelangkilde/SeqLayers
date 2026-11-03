#!/bin/bash
# Mac watcher: per machine (ip tag), when its rescoring diagnostics are done, copy runs/rescore back, append its
# ceiling.csv rows here, and shut it down 5 minutes later (user's standing request).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
watch1() {
  # only a marker newer than this watcher counts (instances are stopped, not terminated, so old markers survive)
  local t0=$(date +%s)
  until $K -n ubuntu@$1 "f=$P/experiments/runs/rescore/done-$2; test -f \$f && [ \$(stat -c %Y \$f) -gt $t0 ]"; do sleep 60; done
  rsync -az -e "$K" ubuntu@$1:$P/experiments/runs/rescore/ $P/experiments/runs/rescore/
  for f in ceiling.csv ceiling_breakdown.csv results.csv; do
    $K -n ubuntu@$1 "grep '\"rescore_' $P/experiments/$f" >> $P/experiments/$f
  done
  $K -n ubuntu@$1 "sudo shutdown -h +5" && echo "$2 ($1): copied back, shutdown scheduled in 5 min, $(date)"
}
watch1 18.116.60.79 awsG &
wait
cat $P/experiments/runs/rescore/summary-awsG.txt
