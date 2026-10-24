#!/bin/bash
# Fallback: power off (EC2 stop) 30 minutes after the analysis passes, the profile and the domain runs finish, if the Mac has not already.
until [ -f /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/rel-analysis/done ] && [ -f /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/rel-analysis/effects_done ] && [ -f /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/profile-softpool/done ] && [ -f /Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers/experiments/runs/domains/done ]; do sleep 60; done
sleep 1800
sudo shutdown -h now
