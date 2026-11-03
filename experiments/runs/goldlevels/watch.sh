#!/bin/bash
# Mac watcher for 3.128.78.18: when the gold levels are done (after the decode profiles), copy runs/profdecode and
# runs/goldlevels back, append their ceiling.csv / results.csv rows here, and shut the instance down 5 minutes later.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
H=ubuntu@3.128.78.18
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
t0=$(date +%s)              # only a marker newer than this watcher counts (instances are stopped, not terminated)
until $K -n $H "f=$P/experiments/runs/goldlevels/done-awsF; test -f \$f && [ \$(stat -c %Y \$f) -gt $t0 ]"; do sleep 120; done
for d in profdecode goldlevels; do
  rsync -az -e "$K" $H:$P/experiments/runs/$d/ $P/experiments/runs/$d/
done
for f in ceiling.csv ceiling_breakdown.csv results.csv; do
  $K -n $H "grep -E '\"(prof|goldlv)_' $P/experiments/$f" >> $P/experiments/$f
done
$K -n $H "sudo shutdown -h +5" && echo "awsF (3.128.78.18): copied back, shutdown scheduled in 5 min, $(date)"
cat $P/experiments/runs/profdecode/summary-awsE.txt $P/experiments/runs/goldlevels/summary-awsF.txt
