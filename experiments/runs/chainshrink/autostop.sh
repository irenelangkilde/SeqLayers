#!/bin/bash
# Fallback: power off (EC2 stop) 30 minutes after this machine's part finishes, if the Mac has not already.
until ls /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/chainshrink/done-aws* >/dev/null 2>&1; do sleep 60; done
sleep 1800
sudo shutdown -h now
