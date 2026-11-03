#!/bin/bash
# Mac watcher: per machine (ip tag), when its fast full decodes are done, copy runs/fastdecode back, append its
# ceiling.csv rows here, and shut it down 5 minutes later (user's standing request).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
watch1() {
  # only a marker newer than this watcher counts (instances are stopped, not terminated, so old markers survive)
  local t0=$(date +%s)
  until $K -n ubuntu@$1 "f=$P/experiments/runs/fastdecode/done-$2; test -f \$f && [ \$(stat -c %Y \$f) -gt $t0 ]"; do sleep 60; done
  rsync -az -e "$K" ubuntu@$1:$P/experiments/runs/fastdecode/ $P/experiments/runs/fastdecode/
  for f in ceiling.csv ceiling_breakdown.csv results.csv; do
    $K -n ubuntu@$1 "grep '\"fast_' $P/experiments/$f" >> $P/experiments/$f
  done
  $K -n ubuntu@$1 "sudo shutdown -h +5" && echo "$2 ($1): copied back, shutdown scheduled in 5 min, $(date)"
}
watch1 3.148.105.132 awsJ &
wait
cat $P/experiments/runs/fastdecode/summary-awsJ.txt
