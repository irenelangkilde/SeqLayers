# Undo sentence splits approximately (merge consecutive predicted sentences inside one gold sentence), attach the extra
# roots to the root with the largest subtree as parataxis.  Usage: python3 -I reattach_check.py gold pred out
import sys
def load(f, words_only):
    sents=[];cur=[]
    for line in open(f):
        line=line.rstrip('\n')
        if not line:
            if cur: sents.append(cur); cur=[]
            continue
        if line.startswith('#'): continue
        c=line.split('\t')
        if words_only and ('-' in c[0] or '.' in c[0]): continue
        cur.append(c)
    if cur: sents.append(cur)
    return sents
def ends(sents):
    out=[];pos=0
    for s in sents:
        for c in s: pos+=len(c[1])
        out.append(pos)
    return out
g=load(sys.argv[1], True); p=load(sys.argv[2], True)
gends=set(ends(g)); pend=ends(p)
merged=[];cur=[]
for k,(s,e) in enumerate(zip(p,pend)):
    off=len(cur)
    for c in s:
        c=list(c); h=int(c[6]); c[6]=str(h+off if h>0 else 0); cur.append(c)
    if e in gends or k==len(p)-1:
        merged.append(cur); cur=[]
nre=0
for s in merged:
    heads=[int(c[6]) for c in s]; roots=[i for i,h in enumerate(heads) if h==0]
    def top(k):
        while heads[k]!=0: k=heads[k]-1
        return k
    if len(roots)>1:
        size={r:sum(1 for k in range(len(s)) if top(k)==r) for r in roots}
        main=max(roots,key=lambda r:size[r])
        for r in roots:
            if r!=main: s[r][6]=str(main+1); s[r][7]='parataxis'; nre+=1
    for i,c in enumerate(s): c[0]=str(i+1)
with open(sys.argv[3],'w') as f:
    for k,s in enumerate(merged):
        f.write('# sent_id = %d\n# text = %s\n'%(k+1,' '.join(c[1] for c in s)))
        for c in s: f.write('\t'.join(c)+'\n')
        f.write('\n')
print('pred sentences',len(p),'-> merged',len(merged),'; extra roots reattached',nre)
