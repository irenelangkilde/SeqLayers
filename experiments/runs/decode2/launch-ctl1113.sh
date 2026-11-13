#!/bin/bash
# Control decode with CoNLL-U output (user 2026-11-13): the best decode (fast kin + nn2, suffixes=3, classes5 levels
# 30/300/3000, relstrikes=1; 0.7453 / 0.6885) rerun on the current code with conllu_out, for the official scorer and as
# the control for the flat-chunk / abbreviation feature runs.  Arg: worker ip.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"; H=ubuntu@$1
experiments/awsname.sh $1
rsync -az -e "$K" src/ $H:$P/src/ && rsync -az -e "$K" experiments/*.jl $H:$P/experiments/ &&
rsync -az -e "$K" experiments/runs/decode2/run-aws.sh $H:$P/experiments/runs/decode2/ &&
$K -f -n $H "cd $P && nohup experiments/runs/decode2/run-aws.sh awsCT ctl1113 suffixes=3,wclasses=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5,wclevels=30+300+3000,relstrikes=1,conllu_out=experiments/runs/decode2/ctl1113 > experiments/runs/decode2/nohup-awsCT.out 2>&1 < /dev/null &" &&
echo "launched ctl1113 on $1"
