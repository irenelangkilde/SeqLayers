# build_pptable.py -- from DepCC (collapsed Stanford dependencies), how often each head word, and each
# word class (clusters/classes5, finest level), takes a prepositional phrase with each preposition
# (arcs head --prep_P--> noun), per occurrence of the head.  Read by SeqLayers.load_pptable.
#   python3 -I build_pptable.py <out.tsv> [min form occurrences=200]
# Output lines (tab-separated):  #total <tokens> | base <P> <PPs> | class <id> <occurrences> <P> <PPs> ... |
#                                form <word> <occurrences> <P> <PPs> ...
import collections, sys
out = sys.argv[1]
minocc = int(sys.argv[2]) if len(sys.argv) > 2 else 200
wc = "/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/clusters/classes5/wordclasses.tsv"
lines = open(wc).read().splitlines()
cls = {l.split("\t")[0]: int(l.split("\t")[2]) for l in lines[1:]}           # finest level
occ, pp, base, total = collections.Counter(), collections.defaultdict(collections.Counter), collections.Counter(), 0
dd = "/Users/irenelangkildegeary2015/Projects/crystallization/juliaNLP/word-embeddings/Udepcc_counts/data"
for f in [dd + "/part-m-000" + x for x in ("09", "34", "60")]:
    sent = []
    def flush():
        global total
        for c in sent:
            occ[c[1].lower()] += 1
            total += 1
            if c[8] != "_":
                hd, rel = c[8].split(":", 1)
                if rel.startswith("prep_"):
                    p = rel[5:].lower()
                    i = int(hd)
                    if 0 <= i < len(sent):
                        pp[sent[i][1].lower()][p] += 1
                        base[p] += 1
    for l in open(f, encoding="utf-8", errors="replace"):
        l = l.rstrip("\n")
        if not l: flush(); sent = []; continue
        c = l.split("\t")
        if len(c) >= 9 and c[0].isdigit(): sent.append(c)
    flush()
preps = [p for p, n in base.most_common() if n >= 100]
cocc, cpp = collections.Counter(), collections.defaultdict(collections.Counter)
for w, n in occ.items():
    if w in cls:
        cocc[cls[w]] += n
        for p, k in pp[w].items(): cpp[cls[w]][p] += k
with open(out, "w") as o:
    print(f"#total\t{total}", file=o)
    for p in preps: print(f"base\t{p}\t{base[p]}", file=o)
    for c in sorted(cocc):
        print("class\t%d\t%d\t" % (c, cocc[c]) + "\t".join(f"{p}\t{cpp[c][p]}" for p in preps if cpp[c][p] > 0), file=o)
    for w, n in occ.items():
        if n >= minocc and pp[w]:
            print("form\t%s\t%d\t" % (w, n) + "\t".join(f"{p}\t{pp[w][p]}" for p in preps if pp[w][p] > 0), file=o)
print(f"{total} tokens; {len(preps)} prepositions with >= 100 PPs; {len(cocc)} classes; forms >= {minocc}: {sum(1 for w, n in occ.items() if n >= minocc and pp[w])}")
