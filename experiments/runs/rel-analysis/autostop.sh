#!/bin/bash
# Fallback: power off (EC2 stop) 30 minutes after the usage pass and the domain follow-ups finish, if the Mac has not already.
until [ -f /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/rel-analysis/done ] && [ -f /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/domains2/done ]; do sleep 60; done
sleep 1800
sudo shutdown -h now
