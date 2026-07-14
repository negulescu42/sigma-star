#!/usr/bin/env python
"""Certified-by-construction training on HotpotQA attention. Frozen spec: certtrain_spec.md.
Fine-tune adapter+QA head with L = L_QA + lambda * mean far-mass in analysed layers.
Sweep lambda; report EM/F1, certified fraction, zero-violation, deletion coupling per condition."""
import json, re, numpy as np, torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

SEED=0; MAXLEN=512; EPS=0.05; TAU0=8.0
LAYERS=[9,10,11,12]; N_TRAIN=1500; N_EVAL=300; EPOCHS=3; LR=1e-4; BATCH=4
LAMBDAS=[0.0,0.5,2.0,8.0]
np.random.seed(SEED); torch.manual_seed(SEED)
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok=AutoTokenizer.from_pretrained('bert-base-uncased')

def norm(s): return re.sub(r'\s+',' ',s.strip().lower())
def hill(w,q):
    w=np.asarray(w,float); s=w.sum()
    if s<=0: return 0.0
    p=w/s
    if q==np.inf: return 1.0/np.max(p)
    return (np.sum(p**q))**(1.0/(1.0-q))

def build_example(ex):
    """Tokenize; return input ids/mask, near/far bool masks (context), gold start/end, or None."""
    q=ex['question']; ans=ex['answer']
    titles=ex['context']['title']; sents=ex['context']['sentences']
    supp=set(zip(ex['supporting_facts']['title'], ex['supporting_facts']['sent_id']))
    ctx=''; sent_spans=[]
    for t,slist in zip(titles,sents):
        for si,s in enumerate(slist):
            st=len(ctx); ctx+=s; en=len(ctx); sent_spans.append((st,en,(t,si) in supp))
    if ans.lower() in ('yes','no'): return None   # span head: drop yes/no
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
    near=np.zeros(len(ids),bool); far=np.zeros(len(ids),bool); supp_present=set()
    for i,(o,sq) in enumerate(zip(off,seqids)):
        t=ids[i]
        if sq==0: near[i]=True; continue
        if t in (tok.cls_token_id,tok.sep_token_id): near[i]=True; continue
        if sq!=1: continue
        c=o[0]; iss2=False
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
    return dict(enc={k:enc[k] for k in ('input_ids','attention_mask','token_type_ids')},
                near=near, far=far, asl=asl, ael=ael, ans=ans, ctx=ctx, off=off, seqids=seqids)

def load_split(split, n, seed):
    ds=load_dataset('hotpot_qa','distractor',split=split)
    idx=np.random.RandomState(seed).permutation(len(ds))
    out=[]
    for i in idx:
        b=build_example(ds[int(i)])
        if b: out.append(b)
        if len(out)>=n: break
    return out

def far_mass_reg(attns, b):
    """Mean outside-evidence attention prob over analysed layers/heads/query-positions (differentiable)."""
    far=torch.tensor(b['far'],device=dev)
    tot=0.0; cnt=0
    for L in LAYERS:
        A=attns[L-1][0]            # [H, seq, seq] post-softmax
        fm=A[:,:,far].sum(-1)      # [H, seq] far mass per (head,query)
        tot=tot+fm.mean(); cnt+=1
    return tot/cnt

def forward_attn(m, b):
    enc={k:v.to(dev) for k,v in b['enc'].items()}
    out=m(**enc, output_attentions=True)
    return out

def eval_condition(m):
    m.eval(); rows=[]; em=0; f1s=[]; n=0
    with torch.no_grad():
        for b in EVAL:
            n+=1
            out=forward_attn(m,b)
            sl=out.start_logits[0]; el=out.end_logits[0]
            si=int(sl.argmax()); ei=int(el.argmax())
            if ei<si: ei=si
            # decode pred span
            ids=b['enc']['input_ids'][0].tolist()
            pred=tok.decode(ids[si:ei+1]).strip()
            gold=b['ans'].strip()
            em+= int(norm(pred)==norm(gold))
            pt=norm(pred).split(); gt=norm(gold).split()
            common=sum((min(pt.count(w),gt.count(w)) for w in set(pt)))
            if len(pt)==0 or len(gt)==0: f1s.append(float(pt==gt))
            else:
                pr=common/len(pt); rc=common/len(gt); f1s.append(0 if pr+rc==0 else 2*pr*rc/(pr+rc))
            # certificate per analysed head/query
            far=b['far']; near=b['near']
            for L in LAYERS:
                A=out.attentions[L-1][0].cpu().numpy()  # [H,seq,seq]
                H=A.shape[0]
                for h in range(H):
                    for qpos in [si]:   # query = predicted start (answer position)
                        p=A[h,qpos]                       # [seq] probs
                        mF=float(p[far].sum())
                        # energies E_i = -tau0*log p_i ; anchor = best near key
                        pnear=p[near]; pfar=p[far]
                        if len(pfar)==0 or pnear.max()<=0: continue
                        EN=-TAU0*np.log(pnear.max()+1e-20)
                        Efar=-TAU0*np.log(pfar+1e-20)
                        Delta=Efar.min()-EN
                        if Delta<=0:
                            rows.append(dict(L=L,h=h,mF=mF,cert=0,Delta=Delta,viol=0)); continue
                        Dq=hill(pfar,2.0)
                        B2=min(1.0, Dq*np.exp(-Delta/TAU0))
                        cert=int(mF<=EPS)
                        viol=int(mF>B2+1e-9)
                        rows.append(dict(L=L,h=h,mF=mF,cert=cert,Delta=float(Delta),viol=viol))
    import pandas as pd
    df=pd.DataFrame(rows)
    pg=df[df.Delta>0]
    return dict(EM=em/n, F1=float(np.mean(f1s)), n=n,
                cert_frac=float(pg.cert.mean()) if len(pg) else 0.0,
                mF_mean=float(df.mF.mean()), viol=int(df.viol.sum()),
                mF_by_layer={int(L):float(df[df.L==L].mF.mean()) for L in LAYERS})

print("loading data...", flush=True)
TRAIN=load_split('train',N_TRAIN,SEED)
EVAL=load_split('validation',N_EVAL,SEED)
print(f"train {len(TRAIN)} eval {len(EVAL)}", flush=True)

results={}
for lam in LAMBDAS:
    torch.manual_seed(SEED); np.random.seed(SEED)
    m=AutoAdapterModel.from_pretrained('bert-base-uncased',attn_implementation='eager').to(dev)
    nm=m.load_adapter('AdapterHub/bert-base-uncased-pf-hotpotqa',source='hf',set_active=True)
    m.set_active_adapters(nm); m.to(dev)
    m.train_adapter(nm)   # freeze backbone, train adapter (+heads)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)
    order=np.random.RandomState(SEED).permutation(len(TRAIN))
    for ep in range(EPOCHS):
        for bi in range(0,len(order),BATCH):
            opt.zero_grad(); loss=0.0
            for j in order[bi:bi+BATCH]:
                b=TRAIN[int(j)]
                out=forward_attn(m,b)
                sl=out.start_logits[0]; el=out.end_logits[0]
                lqa=F.cross_entropy(sl.unsqueeze(0),torch.tensor([b['asl']],device=dev))+\
                    F.cross_entropy(el.unsqueeze(0),torch.tensor([b['ael']],device=dev))
                reg=far_mass_reg(out.attentions,b) if lam>0 else torch.tensor(0.0,device=dev)
                loss=loss+(lqa+lam*reg)/BATCH
            loss.backward(); opt.step()
    res=eval_condition(m)
    results[str(lam)]=res
    print(f"lambda={lam}: EM={res['EM']:.3f} F1={res['F1']:.3f} cert_frac={res['cert_frac']:.4f} "
          f"mF={res['mF_mean']:.3f} viol={res['viol']}", flush=True)
    del m; torch.cuda.empty_cache()

json.dump(results, open('certtrain.json','w'), indent=2)
print(json.dumps(results,indent=2))
