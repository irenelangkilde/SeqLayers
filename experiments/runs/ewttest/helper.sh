#!/bin/bash
# Second half of the overnight EWT test run (user 2026-11-14, option 2), replacing the end of orchestrate.sh: when the
# first worker finishes shard 1/2 it helps with shard 2/2 -- smallest docs first, skipping the docs the second worker
# (biggest first) has finished, whose markers this script copies over every few minutes.  Once every shard-2 doc has a
# prediction on one machine or the other, both runs stop; per-doc files come back, both machines shut down, and the
# predictions are assembled in test-file order (assemble.jl) and scored.  Args: first ip, second ip.
P=/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/SeqLayers
cd $P
K="ssh -i $HOME/.ssh/cpu-worker.pem -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
F=ubuntu@$1; S=ubuntu@$2
OUT=experiments/runs/ewttest
LOG=$OUT/orchestrate.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
for H in $F $S; do
  rsync -az -e "$K" experiments/dev_eval.jl $H:$P/experiments/ && rsync -az -e "$K" $OUT/run-aws.sh $H:$P/$OUT/ || { say "helper: sync to $H failed"; exit 1; }
done
say "helper: armed (code synced)"
done_on() { $K -n $1 "ls $P/$OUT/ | grep -E '^$2-doc' | sed -E 's/^$2-doc([0-9a-f]+)\.conllu/\1/'"; }
until $K -n $F "test -f $P/$OUT/done-first"; do sleep 120; done
rsync -az -e "$K" $F:$P/$OUT/ $P/$OUT/ && say "first: shard 1/2 done, copied back"
$K -n $F "mkdir -p $P/$OUT/skip" && for h in $(done_on $S second); do echo "doc$h"; done | $K $F "cd $P/$OUT/skip && xargs -r touch"
$K -f -n $F "cd $P && nohup $OUT/run-aws.sh help 2/2 0 'docorder=asc skipdir=$OUT/skip' > $OUT/nohup-help.out 2>&1 < /dev/null &" && say "first: helping with shard 2/2 (smallest first)"
while true; do
  sleep 180
  sd=$(done_on $S second); hd=$(done_on $F help)
  for h in $sd; do echo "doc$h"; done | $K $F "cd $P/$OUT/skip && xargs -r touch"
  left=$(printf "%s\n%s\n" "$sd" "$hd" | grep -v '^$' | sort -u | comm -23 <(sort -u $OUT/shard2-docs.txt) -)
  [ -z "$left" ] && { say "shard 2/2: every doc covered ($(echo $sd | wc -w) second, $(echo $hd | wc -w) helper)"; break; }
  $K -n $S "test -f $P/$OUT/done-second" && { say "second: finished shard 2/2 on its own"; break; }
done
for H in $F $S; do $K -n $H "pkill -f 'experiments/dev_eval.jl'; sleep 5; pkill -f julia" ; done
say "runs stopped"
for H in $F $S; do rsync -az -e "$K" $H:$P/$OUT/ $P/$OUT/; $K -n $H "sudo shutdown -h +5"; done
say "copied back; both machines shut down in 5 min"
export PATH=$HOME/.juliaup/bin:$PATH
julia --project=experiments $OUT/assemble.jl >> $LOG 2>&1 &&
python3 -I ~/Projects/data/unidep/ud-test-v2.0-conll2017/evaluation_script/conll17_ud_eval.py -v $OUT/ewttest-gold.conllu $OUT/ewttest-pred.conllu > $OUT/ewttest-official.txt 2>&1
say "scored: $(grep -E '^(UAS|LAS)' $OUT/ewttest-official.txt | tr -s ' ' | tr '\n' ';')"
touch $OUT/done-all
