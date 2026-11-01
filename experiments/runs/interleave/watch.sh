#!/bin/bash
# Mac watcher: per machine (ip tag), when its interleave runs are done, copy runs/interleave back, append its
# ceiling.csv rows here, and shut it down 5 minutes later (user's standing request).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
watch1() {
  until $K -n ubuntu@$1 "test -f $P/experiments/runs/interleave/done-$2"; do sleep 60; done
  rsync -az -e "$K" ubuntu@$1:$P/experiments/runs/interleave/ $P/experiments/runs/interleave/
  for f in ceiling.csv ceiling_breakdown.csv; do
    $K -n ubuntu@$1 "grep '\"il_' $P/experiments/$f" >> $P/experiments/$f
  done
  $K -n ubuntu@$1 "sudo shutdown -h +5" && echo "$2 ($1): copied back, shutdown scheduled in 5 min, $(date)"
}
watch1 3.144.122.251 awsA &
watch1 3.141.30.14 awsB &
wait
cat $P/experiments/runs/interleave/summary-awsA.txt $P/experiments/runs/interleave/summary-awsB.txt
