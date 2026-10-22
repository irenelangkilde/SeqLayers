#!/bin/bash
# Fallback: power off (EC2 stop) 30 minutes after the gate calibration is done.
sleep 120
until [ -z "$(pgrep -f experiments/gate_stats.jl)" ]; do sleep 60; done
sleep 1800
sudo shutdown -h now
