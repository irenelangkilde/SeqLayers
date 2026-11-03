#!/bin/bash
# Copy the labelled gap trace back, then shut the instance down 5 minutes later.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
H=ubuntu@$1
t0=$(date +%s)
until $K -n $H "f=$P/experiments/runs/gaptrace/done-gap2; test -f \$f && [ \$(stat -c %Y \$f) -gt $t0 ]"; do sleep 60; done
rsync -az -e "$K" $H:$P/experiments/runs/gaptrace/gap2-kinnn2.log $P/experiments/runs/gaptrace/
$K -n $H "sudo shutdown -h +5" && echo "$1: copied back, shutdown scheduled in 5 min, $(date)"
