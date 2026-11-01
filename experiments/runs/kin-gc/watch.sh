#!/bin/bash
# Mac watcher: when the AWS kinship (grandchild) runs are done, copy runs/kin back, append their ceiling.csv rows here, and shut the
# instance down 5 minutes later (user's request 2026-10-31).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
H=ubuntu@18.117.151.48
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
until $K -n $H "test -f $P/experiments/runs/kin-gc/done-aws"; do sleep 60; done
rsync -az -e "$K" $H:$P/experiments/runs/kin-gc/ $P/experiments/runs/kin-gc/
for f in ceiling.csv ceiling_breakdown.csv; do
  $K -n $H "grep '\"kingc_' $P/experiments/$f" >> $P/experiments/$f
done
$K -n $H "sudo shutdown -h +5" && echo "shutdown scheduled in 5 min, $(date)"
cat $P/experiments/runs/kin-gc/summary-aws.txt
