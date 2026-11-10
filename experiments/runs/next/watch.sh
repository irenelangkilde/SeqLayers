#!/bin/bash
# Mac watcher (ip, marker path relative to SeqLayers, row pattern, run dirs...): waits for the marker (newer than this
# watcher), copies the run dirs back, appends the csv rows whose note matches the pattern (an ERE, e.g.
# '"decode2_(a|b)"'), then shuts the instance down 5 minutes later (user's standing request).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
IP=$1; MARK=$2; PAT=$3; shift 3
t0=$(date +%s)
until $K -n ubuntu@$IP "f=$P/$MARK; test -f \$f && [ \$(stat -c %Y \$f) -gt $t0 ]"; do sleep 60; done
for d in "$@"; do rsync -az -e "$K" ubuntu@$IP:$P/$d/ $P/$d/; done
for f in ceiling.csv ceiling_breakdown.csv results.csv; do
  $K -n ubuntu@$IP "grep -E '$PAT' $P/experiments/$f" >> $P/experiments/$f
done
$K -n ubuntu@$IP "sudo shutdown -h +5" && echo "$IP: copied back, shutdown scheduled in 5 min, $(date)"
