#!/bin/bash
# Machine B (2026-11-10): ceiling forkthin=400 + relstrikes=2 on kin + nn2, then the decode suffixes=3 + word classes +
# relstrikes=1.  Marks experiments/runs/next/done-awsB2.
cd /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
experiments/runs/next/run-seq.sh awsB2 \
  "experiments/runs/relstrikes/run-aws.sh awsF fk400s2=forkcap=16,forkcomb=avg,forkthin=400,relstrikes=2" \
  "experiments/runs/decode2/run-aws.sh awsD1 sfx3wcrs1 suffixes=3,wclasses=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5,wclevels=30+300,relstrikes=1"
