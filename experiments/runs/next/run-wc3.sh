#!/bin/bash
# wc3 decode test (user 2026-11-11), one after another on one machine: the fast kin + nn2 decode with suffixes=3 + word
# classes 30/300/3000 + relstrikes=1 -- control, + wc3=lex, + wc3=clu (with oclass instead: 0.7436 / 0.6878 in 3.6 h).
# Marks experiments/runs/next/done-awsW3.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
experiments/runs/next/run-seq.sh awsW3 \
  "experiments/runs/decode2/run-aws.sh awsW3a wc3ctl suffixes=3,wclasses=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5,wclevels=30+300+3000,relstrikes=1" \
  "experiments/runs/decode2/run-aws.sh awsW3b wc3lex suffixes=3,wclasses=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5,wclevels=30+300+3000,relstrikes=1,wc3=lex" \
  "experiments/runs/decode2/run-aws.sh awsW3c wc3clu suffixes=3,wclasses=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5,wclevels=30+300+3000,relstrikes=1,wc3=clu"
