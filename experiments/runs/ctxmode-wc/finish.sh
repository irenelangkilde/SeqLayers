#!/bin/bash
# run.sh lost its closing wait (an empty block it ended with was a syntax error): wait for its two run
# groups (PIDs 2427, 2428), then collect the summary and mark it done.
while kill -0 2427 2>/dev/null || kill -0 2428 2>/dev/null; do sleep 60; done
cat /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/ctxmode-wc/part*.txt > /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/ctxmode-wc/summary.txt
touch /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/ctxmode-wc/done
