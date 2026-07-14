#!/usr/bin/env python
"""Controls on the CANONICAL lambda=0 reader (peer unification): task vs random vs top-decile
far partitions, example-level bootstrap CIs, paired task-random gap. Same reader as the audit."""
import json, re, numpy as np, torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

MAXLEN=512; EPS=0.05; TAU0=8.0; LAYERS=[9,10,11,12]
N_TRAIN=1500; EPOCHS=3; LR=1e-4; BATCH=4; SEED=0
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok=AutoTokenizer.from_pretrained('bert-base-uncased')
def norm(s): return re.sub(r'\s+',' ',s.strip().lower())
def hill(w,q):
    w=np.asarray(w,float); s=w.sum()
    if s<=0: return 0.0
    p=w/s
    return 1.0/np.max(p) if q==np.inf else (np.sum(p**q))**(1.0/(1.0-q))

def build_example(ex):
    q=ex['question']; ans=ex['answer']
    titles=ex['context']['title']; sents=ex['context']['sentences']
    supp=set(zip(ex['supporting_facts']['title'], ex['supporting_facts']['sent_id']))
    ctx=''; sent_spans=[]
    for t,slist in zip(titles,sents):
        for si,s in enumerate(slist):
            st=len(ctx); ctx+=s; en=len(ctx); sent_spans.append((st,en,(t,si) in supp))
    if ans.lower() in ('yes','no'): return None
    enc=tok(q,ctx,truncation='only_second',max_length=MAXLEN,return_offsets_mapping=True,return_tensors='pt')
    off=enc['offset_mapping'][0].tolist(); ids=enc['input_ids'][0].tolist(); seqids=enc.sequence_ids(0)
    pos=ctx.lower().find(ans.lower())
    if pos<0: return None
    aend=pos+len(ans); asl=ael=None
    for i,(o,sq) in enumerate(zip(off,seqids)):
        if sq!=1 or o[1]<=o[0]: continue
        if o[0]<=pos<o[1] and asl is None: asl=i
        if o[0]<aend<=o[1]: ael=i
    if asl is None or ael is None or ael<asl: return None
    near=np.zeros(len(ids),bool); far=np.zeros(len(ids),bool); supp_present=set(); ctxmask=np.zeros(len(ids),bool)
    for i,(o,sq) in enumerate(zip(off,seqids)):
        t=ids[i]
        if sq==0: near[i]=True; continue
        if t in (tok.cls_token_id,tok.sep_token_id): near[i]=True; continue
        if sq!=1: continue
        ctxmask[i]=True; c=o[0]; iss2=False
        for k,(st,en,iss) in enumerate(sent_spans):
            if st<=c<en:
                iss2=iss
                if iss: supp_present.add(k)
                break
        if iss2: near[i]=True
        else: far[i]=True
    for i,t in enumerate(ids):
        if t in (tok.cls_token_id,tok.sep_token_id): near[i]=True; far[i]=False
    n_supp=sum(1 for (_,_,iss) in sent_spans if iss)
    if len(supp_present)<n_supp: return None
    if far.sum()==0 or near.sum()==0: return None
    return dict(enc={k:enc[k].to(dev) for k in ('input_ids','attention_mask','token_type_ids')},
                ids=ids, near=near, far=far, ctxmask=ctxmask, asl=asl, ael=ael, ans=ans)

def collect(ds,n,seed):
    idx=np.random.RandomState(seed).permutation(len(ds)); out=[]
    for i in idx:
        b=build_example(ds[int(i)])
        if b: out.append(b)
        if n and len(out)>=n: break
    return out

print("loading...",flush=True)
TRAIN=collect(load_dataset('hotpot_qa','distractor',split='train'),N_TRAIN,0)
DEVALL=collect(load_dataset('hotpot_qa','distractor',split='validation'),None,0)
TEST=DEVALL[len(DEVALL)//5:]
print(f"train {len(TRAIN)} test {len(TEST)}",flush=True)

# retrain canonical lambda=0 seed=0 reader (deterministic, matches audit)
torch.manual_seed(SEED); np.random.seed(SEED)
m=AutoAdapterModel.from_pretrained('bert-base-uncased',attn_implementation='eager').to(dev)
nm=m.load_adapter('AdapterHub/bert-base-uncased-pf-hotpotqa',source='hf',set_active=True)
m.set_active_adapters(nm); m.to(dev); m.train_adapter(nm)
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=LR)
order=np.random.RandomState(SEED).permutation(len(TRAIN)); m.train()
for ep in range(EPOCHS):
    for bi in range(0,len(order),BATCH):
        opt.zero_grad(); loss=0.0
        for j in order[bi:bi+BATCH]:
            b=TRAIN[int(j)]
            out=m(**b['enc'])
            loss=loss+(F.cross_entropy(out.start_logits,torch.tensor([b['asl']],device=dev))+
                       F.cross_entropy(out.end_logits,torch.tensor([b['ael']],device=dev)))/BATCH
        loss.backward(); opt.step()
m.eval()

def cov_for_partition(kind, rng):
    """Return per-example cert fraction (q=2) under a far partition. kind in {task,random,topdecile}."""
    per_ex=[]
    with torch.no_grad():
        for b in TEST:
            out=m(**b['enc'],output_attentions=True)
            si=int(out.start_logits[0].argmax())
            near0=b['near']; far0=b['far']; ctx=b['ctxmask']
            certs=[]
            for L in LAYERS:
                A=out.attentions[L-1][0].cpu().numpy()
                for h in range(A.shape[0]):
                    p=A[h,si]
                    if kind=='task':
                        near,far=near0,far0
                    elif kind=='random':
                        nf=int(far0.sum()); ci=np.where(ctx)[0]
                        if len(ci)<nf or nf==0: certs.append(0); continue
                        fsel=rng.choice(ci,nf,replace=False)
                        far=np.zeros_like(far0); far[fsel]=True
                        near=~far; near[far]=False
                        near=np.zeros_like(near0); near[:]=False
                        for i in range(len(p)):
                            if not far[i]: near[i]=True
                    else:  # topdecile: far = bottom 90% logits among context, near = top 10%
                        ci=np.where(ctx)[0]
                        if len(ci)==0: certs.append(0); continue
                        thr=np.quantile(p[ci],0.9)
                        far=np.zeros_like(far0); near=np.zeros_like(near0)
                        for i in ci: 
                            if p[i]<thr: far[i]=True
                            else: near[i]=True
                        near[~ctx]=True; far[~ctx]=False
                    pf=p[far]
                    if pf.sum()<=0 or near.sum()==0: certs.append(0); continue
                    lp=np.log(np.clip(p,1e-30,None))
                    dtau=float(lp[near].max()-lp[far].max())
                    if dtau<=0: certs.append(0); continue
                    B2=min(1.0,hill(pf,2)*np.exp(lp[far].max()-lp[near].max()))
                    certs.append(int(B2<=EPS))
            per_ex.append(np.mean(certs) if certs else 0.0)
    return np.array(per_ex)

rng=np.random.RandomState(0)
task=cov_for_partition('task',rng)
rand=cov_for_partition('random',rng)
topd=cov_for_partition('topdecile',rng)

def boot(x,n=2000):
    r=np.random.RandomState(0); m_=[]
    for _ in range(n): m_.append(x[r.randint(0,len(x),len(x))].mean())
    return float(np.mean(x)), float(np.percentile(m_,2.5)), float(np.percentile(m_,97.5))
def boot_gap(a,b,n=2000):
    r=np.random.RandomState(0); g=[]
    for _ in range(n):
        idx=r.randint(0,len(a),len(a)); g.append(a[idx].mean()-b[idx].mean())
    return float((a-b).mean()), float(np.percentile(g,2.5)), float(np.percentile(g,97.5))

res=dict(reader="canonical lambda=0 seed=0", n_test=len(TEST),
    task_cov=boot(task), random_cov=boot(rand), topdecile_cov=boot(topd),
    task_minus_random_gap=boot_gap(task,rand))
json.dump(res,open('hotpot_unified_controls.json','w'),indent=2)
print(json.dumps(res,indent=2))
