#!/bin/bash
# run.sh lost its closing wait (an empty block it ended with was a syntax error): wait for its two run
# groups (PIDs 2301, 2302), then collect the summary and mark it done.
while kill -0 2301 2>/dev/null || kill -0 2302 2>/dev/null; do sleep 60; done
cat /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/ctxmode/part*.txt > /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/ctxmode/summary.txt
touch /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/ctxmode/done
