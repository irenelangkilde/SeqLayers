# pp_counts.py -- how often a prepositional phrase (preposition P) attaches to a given head word, in
# the UD training treebanks and in DepCC (collapsed Stanford dependencies, prep_P).
#   python3 -I pp_counts.py   (paths below)
import collections, glob, re, sys
HEADS = {"run": {"run", "ran", "runs", "running"}, "item": {"item", "items"}}
PREPS = ("in", "on")
corpus_list = "/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/word-embeddings_3/data/input/prague/training_corpus.txt"
files = [l.strip() for l in open(corpus_list) if l.strip() and not l.startswith("#")]

def ud_counts():
    occ, pp = collections.Counter(), collections.Counter()
    for f in files:
        sent = []
        def flush():
            byid = {c[0]: c for c in sent}
            for c in sent:
                for h, forms in HEADS.items():
                    if c[1].lower() in forms: occ[h] += 1
                # P --case--> N --(obl|nmod)--> W
                if c[7] == "case" and c[1].lower() in PREPS:
                    n = byid.get(c[6])
                    if n is None: continue
                    w = byid.get(n[6])
                    if w is None: continue
                    for h, forms in HEADS.items():
                        if w[1].lower() in forms: pp[(h, c[1].lower(), n[7].split(":")[0])] += 1
        for l in open(f, encoding="utf-8"):
            l = l.rstrip("\n")
            if not l: flush(); sent = []; continue
            if l.startswith("#"): continue
            c = l.split("\t")
            if re.match(r"^\d+$", c[0]): sent.append(c)
        flush()
    return occ, pp

def depcc_counts():
    occ, pp = collections.Counter(), collections.Counter()
    base = "/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/word-embeddings/Udepcc_counts/data"
    for f in [base + "/part-m-000" + x for x in ("09", "34", "60")]:
        sent = []
        def flush():
            for c in sent:
                for h, forms in HEADS.items():
                    if c[1].lower() in forms: occ[h] += 1
                if c[8] != "_":
                    hd, rel = c[8].split(":", 1)
                    for p in PREPS:
                        if rel == "prep_" + p:
                            i = int(hd)
                            if 0 <= i < len(sent):
                                for h, forms in HEADS.items():
                                    if sent[i][1].lower() in forms: pp[(h, p)] += 1
        for l in open(f, encoding="utf-8", errors="replace"):
            l = l.rstrip("\n")
            if not l: flush(); sent = []; continue
            c = l.split("\t")
            if len(c) >= 9 and c[0].isdigit(): sent.append(c)
        flush()
    return occ, pp

occ, pp = ud_counts()
print("UD training treebanks (full train files):")
for h in HEADS:
    for p in PREPS:
        n = sum(v for (hh, pp_, rel), v in pp.items() if hh == h and pp_ == p)
        rels = ", ".join(f"{rel} {v}" for (hh, pp_, rel), v in sorted(pp.items()) if hh == h and pp_ == p)
        print(f"  {h:5s} + {p:3s}: {n:4d}  of {occ[h]} occurrences ({100*n/max(occ[h],1):.2f} per 100)   [{rels}]")
occ, pp = depcc_counts()
print("DepCC (3 chunks, 40.6M tokens, collapsed dependencies):")
for h in HEADS:
    for p in PREPS:
        n = pp[(h, p)]
        print(f"  {h:5s} + {p:3s}: {n:6d}  of {occ[h]} occurrences ({100*n/max(occ[h],1):.2f} per 100)")
