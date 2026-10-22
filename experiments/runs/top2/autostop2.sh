#!/bin/bash
# Fallback: power off (EC2 stop) 30 minutes after the max-min run finishes.
until [ -f /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/top2/done2 ]; do sleep 60; done
sleep 1800
sudo shutdown -h now
