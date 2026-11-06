#!/bin/bash
# Launch one half of the thin-step run on a worker: ip, tag, stepmin | forkthin.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"; H=ubuntu@$1
experiments/awsname.sh $1
$K -n $H "free -g | sed -n 2p; pgrep -fa julia | head -2; ls -d $P/experiments/work/*htrank-kin-hkinleftright-nn2 >/dev/null && mkdir -p $P/experiments/runs/thin && echo ok" &&
rsync -az -e "$K" src/ $H:$P/src/ && rsync -az -e "$K" experiments/*.jl $H:$P/experiments/ && rsync -az -e "$K" experiments/runs/thin/ $H:$P/experiments/runs/thin/ &&
$K -f -n $H "cd $P && nohup experiments/runs/thin/run-aws.sh $2 $3 > experiments/runs/thin/nohup-$2.out 2>&1 < /dev/null &" && echo "launched $3 on $1"
