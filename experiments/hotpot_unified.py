#!/usr/bin/env python
"""Unified HotpotQA attention experiment (peer-audit revision).
ONE task-performing reader (continued-trained, matched lambda=0 baseline) drives the
diagnostic, deletion analysis, controls AND the training sweep. Full certificate audit.
Reports: positive-gap rate, empirical compliance Pr(m_F<=eps), analytic certified coverage,
efficiency (cert among compliant), bound tightness B/m_F, deletion sharpness (delO vs 2*m_F*B),
example/layer coverage, correct-vs-incorrect stratification, answer-in-support fraction.
Training: 3 seeds x 4 lambda, full retained dev eval, val/test split for lambda selection."""
import json, re, numpy as np, torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

MAXLEN=512; EPS=0.05; TAU0=8.0; LAYERS=[9,10,11,12]
N_TRAIN=1500; EPOCHS=3; LR=1e-4; BATCH=4
SEEDS=[0,1,2]; LAMBDAS=[0.0,0.5,2.0,8.0]
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok=AutoTokenizer.from_pretrained('bert-base-uncased')

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

def build_example(ex):
    q=ex['question']; ans=ex['answer']
    titles=ex['context']['title']; sents=ex['context']['sentences']
    supp=set(zip(ex['supporting_facts']['title'], ex['supporting_facts']['sent_id']))
    ctx=''; sent_spans=[]
    for t,slist in zip(titles,sents):
        for si,s in enumerate(slist):
            st=len(ctx); ctx+=s; en=len(ctx); sent_spans.append((st,en,(t,si) in supp))
    if ans.lower() in ('yes','no'): return None    # span head: drop yes/no
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
    tok_in_supp=np.zeros(len(ids),bool)
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
        if iss2: near[i]=True; tok_in_supp[i]=True
        else: far[i]=True
    for i,t in enumerate(ids):
        if t in (tok.cls_token_id,tok.sep_token_id): near[i]=True; far[i]=False
    n_supp=sum(1 for (_,_,iss) in sent_spans if iss)
    if len(supp_present)<n_supp: return None
    if far.sum()==0 or near.sum()==0: return None
    return dict(enc={k:enc[k].to(dev) for k in ('input_ids','attention_mask','token_type_ids')},
                ids=ids, near=near, far=far, tok_in_supp=tok_in_supp,
                asl=asl, ael=ael, ans=ans)

print("loading data...", flush=True)
ds_tr=load_dataset('hotpot_qa','distractor',split='train')
ds_dv=load_dataset('hotpot_qa','distractor',split='validation')
def collect(ds, n, seed):
    idx=np.random.RandomState(seed).permutation(len(ds)); out=[]
    for i in idx:
        b=build_example(ds[int(i)])
        if b: out.append(b)
        if n and len(out)>=n: break
    return out
TRAIN=collect(ds_tr,N_TRAIN,0)
DEVALL=collect(ds_dv,None,0)   # ALL retained dev (no arbitrary N=300 cap)
nval=len(DEVALL)//5
VAL=DEVALL[:nval]; TEST=DEVALL[nval:]
print(f"train {len(TRAIN)} dev_all {len(DEVALL)} val {len(VAL)} test {len(TEST)}", flush=True)

def new_reader():
    m=AutoAdapterModel.from_pretrained('bert-base-uncased',attn_implementation='eager').to(dev)
    nm=m.load_adapter('AdapterHub/bert-base-uncased-pf-hotpotqa',source='hf',set_active=True)
    m.set_active_adapters(nm); m.to(dev); m.train_adapter(nm)
    return m,nm

def far_reg(attns,b):
    far=torch.tensor(b['far'],device=dev); tot=0.0
    for L in LAYERS:
        A=attns[L-1][0]; tot=tot+A[:,:,far].sum(-1).mean()
    return tot/len(LAYERS)

def train_reader(lam,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m,nm=new_reader()
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=LR)
    order=np.random.RandomState(seed).permutation(len(TRAIN))
    m.train()
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

def cert_rows(m, examples, want_deletion=False):
    """Per analysed head/query-pos: m_F, gap, analytic bound, compliance, deletion (optional)."""
    rows=[]; em=0; f1s=[]
    # value hooks for deletion
    vcap={}
    hks=[]
    if want_deletion:
        NHEAD=m.config.num_attention_heads; HDIM=m.config.hidden_size//NHEAD
        def mk(li0):
            def hk(mod,i,o): vcap[li0]=o.detach()
            return hk
        for L in LAYERS: hks.append(m.bert.encoder.layer[L-1].attention.self.value.register_forward_hook(mk(L-1)))
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
                if want_deletion:
                    Vh=vcap[L-1][0].cpu().numpy().reshape(-1,NHEAD,HDIM).transpose(1,0,2)
                for h in range(H):
                    p=A[h,si]; pf=p[far]
                    mF=float(pf.sum())
                    lp=np.log(np.clip(p,1e-30,None))
                    dtau=float(lp[near].max()-lp[far].max())
                    expt=np.exp(lp[far].max()-lp[near].max())
                    B2=min(1.0,hill(pf,2)*expt) if dtau>0 else 1.0
                    r=dict(L=L,h=h,mF=mF,Delta=dtau,B2=B2,posgap=int(dtau>0),
                           compliant=int(mF<=EPS), cert2=int(dtau>0 and B2<=EPS),
                           correct=correct, ans_in_supp=ans_in_supp)
                    if want_deletion:
                        Vhh=Vh[h]; O=p@Vhh; pn=p.copy(); pn[far]=0; sn=pn.sum()
                        On=(pn/sn)@Vhh if sn>0 else O*0
                        delO=float(np.linalg.norm(O-On)); Bmax=float(np.linalg.norm(Vhh,axis=1).max())
                        r['delO']=delO; r['relpert']=delO/(np.linalg.norm(O)+1e-12)
                        r['del_bound']=2.0*mF*Bmax
                    rows.append(r)
    for hk in hks: hk.remove()
    import pandas as pd
    return pd.DataFrame(rows), em/len(examples), float(np.mean(f1s))

# ---------- Phase A: training sweep, 3 seeds x 4 lambda, full-dev EM/F1 + coverage ----------
import pandas as pd
sweep={}
canonical=None
for lam in LAMBDAS:
    per_seed=[]
    for sd in SEEDS:
        m=train_reader(lam,sd)
        df,em,ff=cert_rows(m, TEST, want_deletion=False)
        pg=df[df.posgap==1]
        row=dict(seed=sd, EM=em, F1=ff, cert2=float(pg.cert2.mean()) if len(pg) else 0.0,
                 mF=float(df.mF.mean()), compliant=float(df.compliant.mean()),
                 posgap=float(df.posgap.mean()))
        per_seed.append(row)
        print(f"lam={lam} seed={sd}: EM={em:.3f} F1={ff:.3f} cert2={row['cert2']:.3f} mF={row['mF']:.3f}",flush=True)
        if lam==0.0 and sd==0: canonical=m   # keep canonical reader for full audit
        else: del m; torch.cuda.empty_cache()
    A=pd.DataFrame(per_seed)
    sweep[str(lam)]=dict(
        EM_mean=float(A.EM.mean()), EM_sd=float(A.EM.std()),
        F1_mean=float(A.F1.mean()), F1_sd=float(A.F1.std()),
        cert2_mean=float(A.cert2.mean()), cert2_sd=float(A.cert2.std()),
        mF_mean=float(A.mF.mean()), compliant_mean=float(A.compliant.mean()),
        posgap_mean=float(A.posgap.mean()), per_seed=per_seed)

# bootstrapped EM difference lambda_best vs lambda=0 (paired over seeds is small; report sd + range)
# ---------- Phase B: FULL AUDIT on canonical lambda=0 seed=0 reader ----------
df,em,ff=cert_rows(canonical, TEST, want_deletion=True)
pg=df[df.posgap==1]; comp=df[df.compliant==1]
audit=dict(
  reader="matched lambda=0 continued-trained (seed 0)",
  n_test=len(TEST), n_rows=len(df), EM=em, F1=ff,
  posgap_rate=float(df.posgap.mean()),
  empirical_compliance=float(df.compliant.mean()),          # Pr(m_F<=eps) directly measured
  analytic_cert2=float(pg.cert2.mean()) if len(pg) else 0.0, # Pr(bound<=eps | posgap)
  analytic_cert2_all=float(df.cert2.mean()),
  efficiency=float(comp.cert2.mean()) if len(comp) else 0.0, # cert among empirically compliant
  viol2=int((pg.mF>pg.B2+1e-9).sum()),
  tightness_B_over_mF_median=float((pg[pg.mF>0].B2/pg[pg.mF>0].mF).median()) if len(pg[pg.mF>0]) else None,
  mF_mean=float(df.mF.mean()), mF_median=float(df.mF.median()),
  del_bound_viol=int((df.delO>df.del_bound+1e-6).sum()),
  del_sharpness_median=float((df[df.delO>0].del_bound/df[df.delO>0].delO).median()),  # bound/observed
  relpert_cert=float(df[df.cert2==1].relpert.mean()) if (df.cert2==1).any() else None,
  relpert_uncert=float(df[df.cert2==0].relpert.mean()),
  cert2_by_layer={int(L):float(df[df.L==L].cert2.mean()) for L in LAYERS},
  mF_by_layer={int(L):float(df[df.L==L].mF.mean()) for L in LAYERS},
  example_cov_stringent=float(df.groupby(df.index//(len(LAYERS)*12)).cert2.min().mean()),
  cert_by_correct={int(c):float(df[df.correct==c].cert2.mean()) for c in [0,1]},
  ans_in_supp_frac=float(df.ans_in_supp.mean()),
)
out=dict(sweep=sweep, audit=audit,
         params=dict(N_TRAIN=len(TRAIN),n_dev_all=len(DEVALL),n_val=len(VAL),n_test=len(TEST),
                     SEEDS=SEEDS,LAMBDAS=LAMBDAS,EPOCHS=EPOCHS,LR=LR,EPS=EPS,LAYERS=LAYERS))
json.dump(out,open('hotpot_unified.json','w'),indent=2)
df.to_parquet('hotpot_unified_audit_rows.parquet')
print("=== AUDIT ==="); print(json.dumps(audit,indent=2))
print("=== SWEEP ==="); print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='per_seed'} for k,v in sweep.items()},indent=2))
