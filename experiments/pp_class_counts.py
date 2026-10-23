# pp_class_counts.py -- DepCC counts of PP attachment (prep_in, prep_on) to the members of the word
# classes (clusters/classes5) that contain given words, at each class level.
#   python3 -I pp_class_counts.py ran item
import collections, sys
words = sys.argv[1:] or ["ran", "item"]
PREPS = ("in", "on")
wc = "/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5/wordclasses.tsv"
lines = open(wc).read().splitlines()
sizes = [int(x) for x in lines[0].split("\t")[1:]]
cls = {}
for l in lines[1:]:
    f = l.split("\t")
    cls[f[0]] = [int(x) for x in f[2:]]
members = {(lv, c): set() for lv in range(len(sizes)) for c in set(v[lv] for v in cls.values())}
for w, v in cls.items():
    for lv in range(len(sizes)):
        members[(lv, v[lv])].add(w)
targets = {(w, lv): cls[w][lv] for w in words if w in cls for lv in range(len(sizes))}
occ, pp = collections.Counter(), collections.Counter()
base = "/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/word-embeddings/Udepcc_counts/data"
want = {key: members[(lv, c)] for key, (lv, c) in ((k, (k[1], v)) for k, v in targets.items())}
for f in [base + "/part-m-000" + x for x in ("09", "34", "60")]:
    sent = []
    def flush():
        for c in sent:
            w = c[1].lower()
            for key, mem in want.items():
                if w in mem: occ[key] += 1
            if c[8] != "_":
                hd, rel = c[8].split(":", 1)
                for p in PREPS:
                    if rel == "prep_" + p:
                        i = int(hd)
                        if 0 <= i < len(sent):
                            hw = sent[i][1].lower()
                            for key, mem in want.items():
                                if hw in mem: pp[(key, p)] += 1
    for l in open(f, encoding="utf-8", errors="replace"):
        l = l.rstrip("\n")
        if not l: flush(); sent = []; continue
        c = l.split("\t")
        if len(c) >= 9 and c[0].isdigit(): sent.append(c)
    flush()
for w in words:
    if w not in cls: print(f"{w}: not in the class table"); continue
    print(f"\n{w}:")
    for lv, k in enumerate(sizes):
        key = (w, lv)
        mem = want[key]
        ex = ", ".join(sorted(mem, key=len)[:8])
        rates = "   ".join(f"+{p}: {pp[(key, p)]:6d} ({100*pp[(key, p)]/max(occ[key],1):.2f} per 100)" for p in PREPS)
        print(f"  level {k:5d}: class of {len(mem):5d} words, {occ[key]:8d} occurrences   {rates}   e.g. {ex}")
