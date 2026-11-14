#!/bin/bash
# Overnight EWT test run on two workers (user 2026-11-13).  Args: first worker ip (has the full-EWT events and their
# warm count cache), second worker ip.  First decodes shard 1/2 at once; second records its own events (prep), and if
# they match first's byte for byte, gets first's count cache (re-signed for its events) before decoding shard 2/2.
# Each machine: results copied back, then shut down.  Then the shards are merged and scored (conll17_ud_eval.py).
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
F=ubuntu@$1; S=ubuntu@$2
OUT=experiments/runs/ewttest
W=experiments/work/train100000-EWT-gold-cdc3db987-wcclasses5-sfx3-rootabs-htrank-kin-hkinleftright-nn2
LOG=$OUT/orchestrate.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
for H in $F $S; do
  rsync -az -e "$K" src/ $H:$P/src/ && rsync -az -e "$K" experiments/*.jl $H:$P/experiments/ &&
  $K -n $H "mkdir -p $P/$OUT" && rsync -az -e "$K" $OUT/run-aws.sh $H:$P/$OUT/ || { say "sync to $H failed"; exit 1; }
done
$K -f -n $F "cd $P && nohup $OUT/run-aws.sh first 1/2 > $OUT/nohup-first.out 2>&1 < /dev/null &" && say "first: shard 1/2 started"
$K -f -n $S "cd $P && nohup $OUT/run-aws.sh prep 1/1 1 > $OUT/nohup-prep.out 2>&1 < /dev/null &" && say "second: recording events (prep)"
# first: copy back and shut down when done (in the background)
( until $K -n $F "test -f $P/$OUT/done-first"; do sleep 120; done
  rsync -az -e "$K" $F:$P/$OUT/ $P/$OUT/ && say "first: done, copied back"; $K -n $F "sudo shutdown -h +5" && say "first: shutdown scheduled" ) &
until $K -n $S "test -f $P/$OUT/done-prep"; do sleep 60; done
say "second: prep done"
fs=$($K -n $F "stat -c %s $P/$W/events/cands.bin $P/$W/events/items.bin" | tr '\n' ' ')
ss=$($K -n $S "stat -c %s $P/$W/events/cands.bin $P/$W/events/items.bin" | tr '\n' ' ')
say "event sizes first: $fs second: $ss"
if [ -n "$fs" ] && [ "$fs" = "$ss" ]; then
  T=$OUT/countcache-tmp; mkdir -p $T
  rsync -az -e "$K" "$F:$P/$W/events/counts_*.bin" $T/ && rsync -az -e "$K" $T/ "$S:$P/$W/events/" &&
  $K -n $S "cd $P && export PATH=\$HOME/.juliaup/bin:\$PATH && julia --project=experiments -e 'include(\"experiments/dev_eval.jl\"); write(\"$W/events/counts_sig.txt\", eventsig(\"$W/events\"))'" &&
  say "second: first's count cache copied and re-signed ($(du -sh $T | cut -f1))"
  rm -rf $T
else
  say "second: event sizes differ -- shard 2/2 runs with its own (cold) count cache"
fi
$K -f -n $S "cd $P && nohup $OUT/run-aws.sh second 2/2 > $OUT/nohup-second.out 2>&1 < /dev/null &" && say "second: shard 2/2 started"
until $K -n $S "test -f $P/$OUT/done-second"; do sleep 120; done
rsync -az -e "$K" $S:$P/$OUT/ $P/$OUT/ && say "second: done, copied back"; $K -n $S "sudo shutdown -h +5" && say "second: shutdown scheduled"
wait
cat $OUT/first-gold.conllu $OUT/second-gold.conllu > $OUT/ewttest-gold.conllu
cat $OUT/first-pred.conllu $OUT/second-pred.conllu > $OUT/ewttest-pred.conllu
python3 -I ~/Projects/data/unidep/ud-test-v2.0-conll2017/evaluation_script/conll17_ud_eval.py -v $OUT/ewttest-gold.conllu $OUT/ewttest-pred.conllu > $OUT/ewttest-official.txt 2>&1
say "scored: $(grep -E '^LAS' $OUT/ewttest-official.txt)"
touch $OUT/done-all
