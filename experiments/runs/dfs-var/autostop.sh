#!/bin/bash
# Fallback: power off (EC2 stop) 30 minutes after the DFS variants and the forward-checking runs finish, if the Mac has not already.
until [ -f /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/dfs-var/done ] && [ -f /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/fwdcheck/done ]; do sleep 60; done
sleep 1800
sudo shutdown -h now
