#!/bin/bash
# Launch part of the suffix tagging run on a worker: ip, tag, configs (name=options, options comma-separated).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"; H=ubuntu@$1; TAG=$2; shift 2
experiments/awsname.sh ${H#ubuntu@}
$K -n $H "free -g | sed -n 2p; pgrep -fa julia | head -2; ls -d $P/experiments/work/*htrank-kin-hkinleftright-nn2 >/dev/null && mkdir -p $P/experiments/runs/suffixtags && echo ok" &&
rsync -az -e "$K" src/ $H:$P/src/ && rsync -az -e "$K" experiments/*.jl $H:$P/experiments/ && rsync -az -e "$K" experiments/runs/suffixtags/ $H:$P/experiments/runs/suffixtags/ &&
$K -f -n $H "cd $P && nohup experiments/runs/suffixtags/run-aws.sh $TAG $* > experiments/runs/suffixtags/nohup-$TAG.out 2>&1 < /dev/null &" && echo "launched $* on ${H#ubuntu@}"
