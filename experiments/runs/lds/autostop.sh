#!/bin/bash
# Fallback: power off (EC2 stop) 30 minutes after the search runs and the context-rule runs finish, if the Mac has not already.
until [ -f /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/lds/done ] && [ -f /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/ctxrules/done ]; do sleep 60; done
sleep 1800
sudo shutdown -h now
