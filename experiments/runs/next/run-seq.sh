#!/bin/bash
# Run several existing run scripts one after another on a worker, then mark done (user 2026-11-10: a ceiling, then a
# decode, on one machine -- a kin + nn2 decode leaves no room beside it).  Args: tag, then commands, each one quoted
# string (a run script with its arguments).  Marks experiments/runs/next/done-<tag>.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
TAG=$1; shift
rm -f experiments/runs/next/done-$TAG
for c in "$@"; do
  echo "=== $(date) $c" >> experiments/runs/next/seq-$TAG.txt
  bash -c "$c"
done
echo "=== all done $(date)" >> experiments/runs/next/seq-$TAG.txt
touch experiments/runs/next/done-$TAG
