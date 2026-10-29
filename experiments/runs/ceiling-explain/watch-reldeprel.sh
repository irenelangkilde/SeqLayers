#!/bin/bash
# Mac watcher: when the relation traces are done, copy the logs back and stop the instance (EC2 stop).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
K="-i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20"
IP=$1
until ssh -n $K ubuntu@$IP "test -f $P/experiments/runs/ceiling-explain/done-reldeprel"; do sleep 120; done
rsync -az -e "ssh $K" --include='detail-rel-*.log' --include='done-reldeprel' --exclude='*' ubuntu@$IP:$P/experiments/runs/ceiling-explain/ $P/experiments/runs/ceiling-explain/
ssh -n -f $K ubuntu@$IP "setsid nohup bash -c 'sleep 540; sudo shutdown -h now' > /dev/null 2>&1 < /dev/null &"
echo "copied back, shutdown scheduled $(date)"
for d in conj parataxis acl advcl; do echo "$d: $(grep -c '^================' $P/experiments/runs/ceiling-explain/detail-rel-$d.log) sentences"; done
