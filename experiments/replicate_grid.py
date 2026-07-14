#!/usr/bin/env python
"""Cross-model x cross-dataset replication of the task-grounded attention certificate.
Grid: {BERT, RoBERTa} x {HotpotQA, 2Wiki}. Each cell: continued-train matched lambda=0
reader + lambda=8 far-mass-regularized reader (1 seed), full certificate audit + deletion
+ task-vs-random control on the lambda=0 reader. Frozen protocol identical to hotpot_unified.py.
"""
import json, re, sys, numpy as np, torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

MAXLEN=512; EPS=0.05; TAU0=8.0; LAYERS=[9,10,11,12]
N_TRAIN=1500; EPOCHS=3; LR=1e-4; BATCH=4; SEED=0
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS={
 "bert":    dict(base="bert-base-uncased", adapter="AdapterHub/bert-base-uncased-pf-hotpotqa", attr="bert"),
 "roberta": dict(base="roberta-base",      adapter="AdapterHub/roberta-base-pf-hotpotqa",      attr="roberta"),
}
DATASETS={
 "hotpot": dict(loader=("hotpot_qa","distractor")),
 "2wiki":  dict(loader=("scholarly-shadows-syndicate/2WikiMultihopQA_with_q_gpt35",None)),
}

def norm(s): return re.sub(r'\s+',' ',s.strip().lower())
def hill(w,q):
    w=np.asarray(w,float); s=w.sum()
    if s<=0: return 0.0
    p=w/s
    return 1.0/np.max(p) if q==np.inf else (np.sum(p**q))**(1.0/(1.0-q))
def f1(pred,gold):
    ps=norm(pred).split(); gs=norm(gold).split()
    if not ps or not gs: return float(ps==gs)
    ov=sum(min(ps.count(w),gs.count(w)) for w in set(ps))
    if ov==0: return 0.0
    p=ov/len(ps); r=ov/len(gs); return 2*p*r/(p+r)

def build_example(ex, tok):
    q=ex['question']; ans=str(ex['answer'])
    ctx_field=ex['context']
    if not isinstance(ctx_field,dict): return None
    # HotpotQA HF uses key 'sentences'; the 2Wiki parquet mirror uses 'content'.
    # Both are parallel to 'title': a list (per doc) of sentence-lists.
    skey='sentences' if 'sentences' in ctx_field else ('content' if 'content' in ctx_field else None)
    if skey is None or 'title' not in ctx_field: return None
    titles=ctx_field['title']; sents=ctx_field[skey]
    sf=ex['supporting_facts']
    if not isinstance(sf,dict): return None
    supp=set(zip(sf['title'], sf['sent_id']))
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
    near=np.zeros(len(ids),bool); far=np.zeros(len(ids),bool); supp_present=set()
    tok_in_supp=np.zeros(len(ids),bool); ctxmask=np.zeros(len(ids),bool)
    spec=set(t for t in (tok.cls_token_id,tok.sep_token_id,tok.bos_token_id,tok.eos_token_id,tok.pad_token_id) if t is not None)
    for i,(o,sq) in enumerate(zip(off,seqids)):
        t=ids[i]
        if sq==0: near[i]=True; continue
        if t in spec: near[i]=True; continue
        if sq!=1: continue
        ctxmask[i]=True; c=o[0]; iss2=False
        for k,(st,en,iss) in enumerate(sent_spans):
            if st<=c<en:
                iss2=iss
                if iss: supp_present.add(k)
                break
        if iss2: near[i]=True; tok_in_supp[i]=True
        else: far[i]=True
    for i,t in enumerate(ids):
        if t in spec: near[i]=True; far[i]=False; ctxmask[i]=False
    n_supp=sum(1 for (_,_,iss) in sent_spans if iss)
    if len(supp_present)<n_supp: return None
    if far.sum()==0 or near.sum()==0: return None
    keys=[k for k in ('input_ids','attention_mask','token_type_ids') if k in enc]
    return dict(enc={k:enc[k].to(dev) for k in keys},
                ids=ids, near=near, far=far, ctxmask=ctxmask, tok_in_supp=tok_in_supp,
                asl=asl, ael=ael, ans=ans)

def collect(ds, n, tok, seed=0):
    idx=np.random.RandomState(seed).permutation(len(ds)); out=[]
    for i in idx:
        try: b=build_example(ds[int(i)], tok)
        except Exception: b=None
        if b: out.append(b)
        if n and len(out)>=n: break
    return out

def collect_slice(ds, idx_order, n, tok):
    """Collect up to n built examples following a supplied index order (disjoint slices safe)."""
    out=[]
    for i in idx_order:
        try: b=build_example(ds[int(i)], tok)
        except Exception: b=None
        if b: out.append(b)
        if n and len(out)>=n: break
    return out

def load_splits(dskey, tok):
    nm,cfg=DATASETS[dskey]["loader"]
    tr=load_dataset(nm,cfg,split="train") if cfg else load_dataset(nm,split="train")
    devcap = None if dskey!="2wiki" else 1200
    try:
        dv=load_dataset(nm,cfg,split="validation") if cfg else load_dataset(nm,split="validation")
        # Separate splits: independent permutations are safe (different underlying data).
        TRAIN=collect_slice(tr, np.random.RandomState(0).permutation(len(tr)), N_TRAIN, tok)
        DEVALL=collect_slice(dv, np.random.RandomState(0).permutation(len(dv)), devcap, tok)
    except Exception:
        # No validation split: carve a DISJOINT held-out index range from train.
        perm=np.random.RandomState(0).permutation(len(tr))
        # over-provision raw indices (retention ~30-60%) so both slices fill without overlap
        cut=min(len(perm)//2, N_TRAIN*4)
        train_idx=perm[:cut]; eval_idx=perm[cut:]  # strictly disjoint
        TRAIN=collect_slice(tr, train_idx, N_TRAIN, tok)
        DEVALL=collect_slice(tr, eval_idx, devcap, tok)  # devcap=None -> uncapped, same as try-branch
    return TRAIN, DEVALL

def make_reader(mkey):
    cfg=MODELS[mkey]
    m=AutoAdapterModel.from_pretrained(cfg["base"],attn_implementation='eager').to(dev)
    nm=m.load_adapter(cfg["adapter"],source='hf',set_active=True)
    m.set_active_adapters(nm); m.to(dev); m.train_adapter(nm)
    return m,nm

def far_reg(attns,b):
    far=torch.tensor(b['far'],device=dev); tot=0.0
    for L in LAYERS:
        A=attns[L-1][0]; tot=tot+A[:,:,far].sum(-1).mean()
    return tot/len(LAYERS)

def train_reader(mkey,lam,TRAIN,seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m,nm=make_reader(mkey)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=LR)
    order=np.random.RandomState(seed).permutation(len(TRAIN)); m.train()
    for ep in range(EPOCHS):
        for bi in range(0,len(order),BATCH):
            opt.zero_grad(); loss=0.0
            for j in order[bi:bi+BATCH]:
                b=TRAIN[int(j)]
                out=m(**b['enc'],output_attentions=True)
                lqa=F.cross_entropy(out.start_logits,torch.tensor([b['asl']],device=dev))+\
                    F.cross_entropy(out.end_logits,torch.tensor([b['ael']],device=dev))
                reg=far_reg(out.attentions,b) if lam>0 else torch.tensor(0.0,device=dev)
                loss=loss+(lqa+lam*reg)/BATCH
            loss.backward(); opt.step()
    m.eval(); return m

def encoder_layers(m,mkey): return getattr(m,MODELS[mkey]["attr"]).encoder.layer

import pandas as pd
def cert_rows(m, mkey, tok, examples, want_deletion=False):
    rows=[]; em=0; f1s=[]; vcap={}; hks=[]
    NHEAD=m.config.num_attention_heads; HDIM=m.config.hidden_size//NHEAD
    if want_deletion:
        def mk(li0):
            def hk(mod,i,o): vcap[li0]=o.detach()
            return hk
        for L in LAYERS: hks.append(encoder_layers(m,mkey)[L-1].attention.self.value.register_forward_hook(mk(L-1)))
    with torch.no_grad():
        for b in examples:
            out=m(**b['enc'],output_attentions=True)
            si=int(out.start_logits[0].argmax()); ei=int(out.end_logits[0].argmax())
            if ei<si: ei=si
            pred=tok.decode(b['ids'][si:ei+1]).strip(); gold=b['ans']
            correct=int(norm(pred)==norm(gold)); em+=correct; f1s.append(f1(pred,gold))
            ans_in_supp=int(b['tok_in_supp'][si]) if si<len(b['tok_in_supp']) else 0
            near=b['near']; far=b['far']
            for L in LAYERS:
                A=out.attentions[L-1][0].cpu().numpy(); H=A.shape[0]
                if want_deletion: Vh=vcap[L-1][0].cpu().numpy().reshape(-1,NHEAD,HDIM).transpose(1,0,2)
                for h in range(H):
                    p=A[h,si]; pf=p[far]; mF=float(pf.sum())
                    lp=np.log(np.clip(p,1e-30,None))
                    dtau=float(lp[near].max()-lp[far].max())
                    B2=min(1.0,hill(pf,2)*np.exp(lp[far].max()-lp[near].max())) if dtau>0 else 1.0
                    r=dict(L=L,h=h,mF=mF,Delta=dtau,B2=B2,posgap=int(dtau>0),
                           compliant=int(mF<=EPS),cert2=int(dtau>0 and B2<=EPS),
                           correct=correct,ans_in_supp=ans_in_supp)
                    if want_deletion:
                        Vhh=Vh[h]; O=p@Vhh; pn=p.copy(); pn[far]=0; sn=pn.sum()
                        On=(pn/sn)@Vhh if sn>0 else O*0
                        delO=float(np.linalg.norm(O-On))
                        r['delO']=delO; r['relpert']=delO/(np.linalg.norm(O)+1e-12)
                        r['del_bound']=2.0*mF*float(np.linalg.norm(Vhh,axis=1).max())
                    rows.append(r)
    for hk in hks: hk.remove()
    return pd.DataFrame(rows), em/max(len(examples),1), float(np.mean(f1s)) if f1s else 0.0

def random_control_cov(m, mkey, tok, examples, rng):
    per=[]
    with torch.no_grad():
        for b in examples:
            out=m(**b['enc'],output_attentions=True)
            si=int(out.start_logits[0].argmax()); ctx=b['ctxmask']; nf=int(b['far'].sum())
            ci=np.where(ctx)[0]; certs=[]
            for L in LAYERS:
                A=out.attentions[L-1][0].cpu().numpy()
                for h in range(A.shape[0]):
                    p=A[h,si]
                    if len(ci)<nf or nf==0: certs.append(0); continue
                    fsel=rng.choice(ci,nf,replace=False)
                    far=np.zeros(len(p),bool); far[fsel]=True; near=~far
                    pf=p[far]
                    if pf.sum()<=0: certs.append(0); continue
                    lp=np.log(np.clip(p,1e-30,None))
                    dtau=float(lp[near].max()-lp[far].max())
                    if dtau<=0: certs.append(0); continue
                    certs.append(int(min(1.0,hill(pf,2)*np.exp(lp[far].max()-lp[near].max()))<=EPS))
            per.append(np.mean(certs) if certs else 0.0)
    return np.array(per)

def boot(x,n=2000):
    r=np.random.RandomState(0); m_=[x[r.randint(0,len(x),len(x))].mean() for _ in range(n)]
    return float(np.mean(x)),float(np.percentile(m_,2.5)),float(np.percentile(m_,97.5))
def boot_gap(a,b,n=2000):
    r=np.random.RandomState(0); g=[]
    for _ in range(n):
        idx=r.randint(0,len(a),len(a)); g.append(a[idx].mean()-b[idx].mean())
    return float((a-b).mean()),float(np.percentile(g,2.5)),float(np.percentile(g,97.5))

SEEDS=[0,1,2]
def ms(xs):
    xs=[x for x in xs if x is not None]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (None,None)

GRID=[(mk,dk) for mk in MODELS for dk in DATASETS]
results={}
for mkey,dkey in GRID:
    cell=f"{mkey}x{dkey}"
    print(f"\n########## CELL {cell} ##########",flush=True)
    tok=AutoTokenizer.from_pretrained(MODELS[mkey]["base"])
    TRAIN,DEVALL=load_splits(dkey,tok)
    nval=len(DEVALL)//5; VAL=DEVALL[:nval]; TEST=DEVALL[nval:]
    print(f"{cell}: train {len(TRAIN)} test {len(TEST)}",flush=True)
    if len(TRAIN)<200 or len(TEST)<100:
        results[cell]={"error":f"too few examples train={len(TRAIN)} test={len(TEST)}"}; continue
    perseed=[]
    for sd in SEEDS:
        # lambda=0 canonical reader: audit + deletion + control
        m0=train_reader(mkey,0.0,TRAIN,sd)
        df,em0,f0=cert_rows(m0,mkey,tok,TEST,want_deletion=True)
        pg=df[df.posgap==1]
        nh=len(LAYERS)*m0.config.num_attention_heads
        task_cov=df.groupby(df.index//nh).cert2.mean().values
        rand=random_control_cov(m0,mkey,tok,TEST,np.random.RandomState(sd))
        gap=boot_gap(task_cov,rand)
        # lambda=8 trained reader
        m8=train_reader(mkey,8.0,TRAIN,sd)
        df8,em8,f8=cert_rows(m8,mkey,tok,TEST,want_deletion=False)
        pg8=df8[df8.posgap==1]
        s=dict(seed=sd, EM=em0, F1=f0,
               posgap_rate=float(df.posgap.mean()),
               empirical_compliance=float(df.compliant.mean()),
               analytic_cert2=float(pg.cert2.mean()) if len(pg) else 0.0,
               viol2=int((pg.mF>pg.B2+1e-9).sum()),
               mF_mean=float(df.mF.mean()),
               del_bound_viol=int((df.delO>df.del_bound+1e-6).sum()),
               relpert_cert=float(df[df.cert2==1].relpert.mean()) if (df.cert2==1).any() else None,
               relpert_uncert=float(df[df.cert2==0].relpert.mean()),
               task_cov=float(task_cov.mean()), random_cov=float(rand.mean()),
               task_minus_random=gap[0], gap_ci=[gap[1],gap[2]],
               EM_lam8=em8, F1_lam8=f8,
               cert2_lam8=float(pg8.cert2.mean()) if len(pg8) else 0.0,
               compliant_lam8=float(df8.compliant.mean()), mF_lam8=float(df8.mF.mean()),
               viol2_lam8=int((pg8.mF>pg8.B2+1e-9).sum()))
        perseed.append(s)
        print(f"  seed {sd}: EM {em0:.3f} cert0 {s['analytic_cert2']:.3f} gap {s['task_minus_random']:.3f} "
              f"cert8 {s['cert2_lam8']:.3f} EM8 {em8:.3f} viol {s['viol2']}/{s['viol2_lam8']}",flush=True)
        del m0,m8; torch.cuda.empty_cache()
    agg={k:ms([s[k] for s in perseed]) for k in
         ["EM","F1","posgap_rate","empirical_compliance","analytic_cert2","mF_mean",
          "relpert_cert","relpert_uncert","task_cov","random_cov","task_minus_random",
          "EM_lam8","cert2_lam8","compliant_lam8","mF_lam8"]}
    agg["n_test"]=len(TEST)
    agg["viol2_total"]=int(sum(s["viol2"] for s in perseed))
    agg["viol2_lam8_total"]=int(sum(s["viol2_lam8"] for s in perseed))
    agg["del_bound_viol_total"]=int(sum(s["del_bound_viol"] for s in perseed))
    agg["gap_ci_all"]=[s["gap_ci"] for s in perseed]
    results[cell]=dict(agg=agg, perseed=perseed)
    print(f"{cell} AGG: EM {agg['EM'][0]:.3f}+-{agg['EM'][1]:.3f}  cert0 {agg['analytic_cert2'][0]:.3f}+-{agg['analytic_cert2'][1]:.3f}  "
          f"gap {agg['task_minus_random'][0]:.3f}+-{agg['task_minus_random'][1]:.3f}  cert8 {agg['cert2_lam8'][0]:.3f}+-{agg['cert2_lam8'][1]:.3f}  "
          f"viol {agg['viol2_total']}/{agg['viol2_lam8_total']}",flush=True)

json.dump(dict(results=results,params=dict(N_TRAIN=N_TRAIN,EPOCHS=EPOCHS,LR=LR,EPS=EPS,
              LAYERS=LAYERS,SEED=SEED,grid=[f"{a}x{b}" for a,b in GRID])),
          open('replicate_grid.json','w'),indent=2)
print("\n=== DONE ===")
print(json.dumps(results,indent=2))
