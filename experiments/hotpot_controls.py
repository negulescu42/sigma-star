#!/usr/bin/env python
"""Controls for task-grounded certification (SI): (a) top-decile-logit endogenous
partition, (b) size-matched RANDOM partition fixed independent of logits.
Reuses the frozen pipeline's model/data; measures far mass + cert coverage + deletion
coupling under each control partition, on the SAME kept examples."""
import json, re, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

SEED=0; N_EVAL=300; MAXLEN=512; EPS=0.05; TAU0=8.0; OVERSAMPLE=12
LAYERS=[9,10,11,12]
np.random.seed(SEED); torch.manual_seed(SEED)
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok=AutoTokenizer.from_pretrained('bert-base-uncased')
m=AutoAdapterModel.from_pretrained('bert-base-uncased', attn_implementation='eager').to(dev).eval()
nm=m.load_adapter('AdapterHub/bert-base-uncased-pf-hotpotqa', source='hf', set_active=True)
m.set_active_adapters(nm); m.to(dev)
NHEAD=m.config.num_attention_heads; HDIM=m.config.hidden_size//NHEAD
enc_layers=m.bert.encoder.layer
_vcap={}
def _mk(li0):
    def hk(md,i,o): _vcap[li0]=o.detach()
    return hk
for L in LAYERS: enc_layers[L-1].attention.self.value.register_forward_hook(_mk(L-1))

def norm(s): return re.sub(r'\s+',' ',s.strip().lower())
def hill(w,q):
    w=np.asarray(w,float); s=w.sum()
    if s<=0: return 0.0
    p=w/s
    return 1.0/np.max(p) if q==np.inf else (np.sum(p**q))**(1.0/(1.0-q))

ds=load_dataset('hotpot_qa','distractor',split='validation')
idx=np.random.RandomState(SEED).permutation(len(ds))[:N_EVAL*OVERSAMPLE]

# reuse build_example logic (near/far from supporting facts) to identify VALID examples + universe
import importlib.util
spec=importlib.util.spec_from_file_location("hc","hotpot_cert.py")
# We can't import (it runs on import); instead re-implement the retention+partition inline.
def build(ex):
    q=ex['question']; ans=ex['answer']
    titles=ex['context']['title']; sents=ex['context']['sentences']
    supp=set(zip(ex['supporting_facts']['title'], ex['supporting_facts']['sent_id']))
    ctx=''; sent_spans=[]
    for t,slist in zip(titles,sents):
        for si,s in enumerate(slist):
            st=len(ctx); ctx+=s; en=len(ctx); sent_spans.append((st,en,(t,si) in supp))
    yn=ans.lower() in ('yes','no')
    enc=tok(q,ctx,truncation='only_second',max_length=MAXLEN,return_offsets_mapping=True,return_tensors='pt')
    off=enc['offset_mapping'][0].tolist(); ids=enc['input_ids'][0].tolist(); sq=enc.sequence_ids(0)
    if not yn:
        pos=ctx.lower().find(ans.lower())
        if pos<0: return None
    near=np.zeros(len(ids),bool); far=np.zeros(len(ids),bool); ctxtok=np.zeros(len(ids),bool)
    supp_present=set()
    for i,(o,s_) in enumerate(zip(off,sq)):
        if s_==0: near[i]=True; continue
        if ids[i] in (tok.cls_token_id,tok.sep_token_id): near[i]=True; continue
        if s_!=1: continue
        ctxtok[i]=True; c=o[0]; iss=False
        for k,(st,en,f) in enumerate(sent_spans):
            if st<=c<en:
                iss=f
                if f: supp_present.add(k)
                break
        if iss: near[i]=True
        else: far[i]=True
    n_supp=sum(1 for(_,_,f) in sent_spans if f)
    if len(supp_present)<n_supp or far.sum()==0 or near.sum()==0: return None
    return dict(enc={k:v.to(dev) for k,v in enc.items() if k!='offset_mapping'},
                ids=ids, near=near, far=far, ctxtok=ctxtok, yn=yn)

def cert_rows(b, part_fn, out, tag, exid):
    with torch.no_grad(): o=m(**b['enc'], output_attentions=True)
    sp=o.start_logits.argmax().item(); ep=o.end_logits.argmax().item()
    qpos=[b['ids'].index(tok.cls_token_id)] if b['yn'] else [sp,ep]
    for li,L in enumerate(LAYERS):
        A=o.attentions[L-1][0].cpu().numpy(); H=A.shape[0]
        V=_vcap[L-1][0].cpu().numpy().reshape(-1,NHEAD,HDIM).transpose(1,0,2)
        for h in range(H):
            Vh=V[h]; Bmax=float(np.linalg.norm(Vh,axis=1).max())
            for qp in qpos:
                p=A[h,qp,:]
                near,far=part_fn(b,p)          # control-specific partition
                if far.sum()==0 or near.sum()==0: continue
                pf=p[far]; m_F=float(pf.sum())
                with np.errstate(divide='ignore'): lp=np.log(np.clip(p,1e-30,None))
                expterm=np.exp(lp[far].max()-lp[near].max()); Delta=float(lp[near].max()-lp[far].max())
                B2=min(1.0,hill(pf,2)*expterm)
                O=p@Vh; pn=p.copy(); pn[far]=0; sn=pn.sum()
                On=(pn/sn)@Vh if sn>0 else O*0
                relp=float(np.linalg.norm(O-On)/(np.linalg.norm(O)+1e-12))
                out.append(dict(tag=tag,ex=exid,L=L,h=h,m_F=m_F,Delta=Delta,B2=B2,
                    cert2=int(Delta>0 and B2<=EPS), relpert=relp,
                    del_ok=int(np.linalg.norm(O-On)<=2*m_F*Bmax+1e-6)))

# control partitions (both defined from CONTEXT tokens only, independent of supporting facts)
def part_topdecile(b,p):
    ctx=b['ctxtok']
    ci=np.where(ctx)[0]
    if len(ci)==0: return b['near'],b['far']
    thr=np.quantile(p[ci],0.9)                 # top-decile logit keys = near
    near=np.zeros_like(ctx,dtype=bool)         # explicit bool dtype
    near[ci[p[ci]>=thr]]=True                  # high-logit context tokens -> near
    near=near | (b['near'] & ~ctx)             # + question/special tokens
    far=ctx & ~near                            # remaining context -> far
    return near,far
_rng=np.random.RandomState(SEED)
def part_random(b,p):
    # size-matched to task-grounded far set, chosen from context tokens independent of logits
    ctx=b['ctxtok']; ci=np.where(ctx)[0]; nfar=int(b['far'].sum())
    nfar=min(nfar,len(ci))
    farsel=_rng.choice(ci,size=nfar,replace=False) if nfar>0 else np.array([],int)
    far=np.zeros_like(ctx); far[farsel]=True
    near=b['near'].copy(); near=near & ~far  # near = original near minus any overlap
    near=near | (ctx & ~far)                 # remaining context -> near
    return near,far

kept=0; rows=[]
for ii in idx:
    b=build(ds[int(ii)])
    if b is None: continue
    kept+=1
    cert_rows(b, part_topdecile, rows, 'topdecile', int(ii))
    cert_rows(b, part_random, rows, 'random', int(ii))
    if kept>=N_EVAL: break

import pandas as pd
cdf=pd.DataFrame(rows)
res={}
for tag in ['topdecile','random']:
    s=cdf[cdf.tag==tag]
    res[tag]=dict(n=len(s), m_F_mean=float(s.m_F.mean()), cert2_frac=float(s.cert2.mean()),
        relpert_cert=float(s[s.cert2==1].relpert.mean()) if (s.cert2==1).any() else None,
        relpert_uncert=float(s[s.cert2==0].relpert.mean()),
        del_viol=int((s.del_ok==0).sum()))
json.dump(res, open('hotpot_controls.json','w'), indent=2)
cdf.to_parquet('hotpot_controls_rows.parquet')
print(json.dumps(res,indent=2))
