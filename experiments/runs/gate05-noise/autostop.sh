#!/bin/bash
# Fallback: power off (EC2 stop) 30 minutes after the run finishes, if the Mac has not already.
until [ -f /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/gate05-noise/done ]; do sleep 60; done
sleep 1800
sudo shutdown -h now
