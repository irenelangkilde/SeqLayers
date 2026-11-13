#!/bin/bash
# Feature decodes against the ctl1113 control (user 2026-11-13): abbrev (tokenizer chunk shape + known abbreviation),
# chunks (base-phrase BIOES), mwe (compound / flat / fixed / goeswith spans), chunks + mwe -- each on the best decode
# (fast kin + nn2, suffixes=3, classes5 30/300/3000, relstrikes=1) with CoNLL-U output.  One decode at a time per machine
# (~70 GB each).  Args: first worker ip (runs mwe, chkmwe after its ctl1113 control), second worker ip (abbrev, chunks).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
B=suffixes=3,wclasses=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5,wclevels=30+300+3000,relstrikes=1
R=experiments/runs/decode2/run-aws.sh
O=conllu_out=experiments/runs/decode2
sync() {
  rsync -az -e "$K" src/ ubuntu@$1:$P/src/ && rsync -az -e "$K" experiments/*.jl ubuntu@$1:$P/experiments/ &&
  rsync -az -e "$K" experiments/runs/decode2/run-aws.sh ubuntu@$1:$P/experiments/runs/decode2/ &&
  rsync -az -e "$K" experiments/runs/next/run-seq.sh ubuntu@$1:$P/experiments/runs/next/
}
experiments/awsname.sh $1; experiments/awsname.sh $2
sync $1 && sync $2 || exit 1
# second worker: abbrev, then chunks
$K -f -n ubuntu@$2 "cd $P && nohup experiments/runs/next/run-seq.sh awsQS \"$R awsAB abbrev1113 $B,abbrev=true,$O/abbrev1113\" \"$R awsCH chunks1113 $B,chunks=true,$O/chunks1113\" > experiments/runs/next/nohup-awsQS.out 2>&1 < /dev/null &" &&
echo "launched abbrev1113, chunks1113 on $2"
# first worker: after the running control (done-awsCT), mwe, then chunks + mwe
$K -f -n ubuntu@$1 "cd $P && nohup bash -c 'until [ -f experiments/runs/decode2/done-awsCT ]; do sleep 60; done; experiments/runs/next/run-seq.sh awsQF \"$R awsMW mwe1113 $B,mwe=true,$O/mwe1113\" \"$R awsCM chkmwe1113 $B,chunks=true,mwe=true,$O/chkmwe1113\"' > experiments/runs/next/nohup-awsQF.out 2>&1 < /dev/null &" &&
echo "queued mwe1113, chkmwe1113 on $1 after ctl1113"
