#!/bin/bash
# Fallback: power off (EC2 stop) 30 minutes after both runs and the gate calibration are done.
until [ -f /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/headconf/done ] && [ -z "$(pgrep -f experiments/gate_stats.jl)" ]; do sleep 60; done
sleep 1800
sudo shutdown -h now
