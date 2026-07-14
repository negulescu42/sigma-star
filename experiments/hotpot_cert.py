#!/usr/bin/env python
"""Task-grounded attention certification on HotpotQA distractor.
Frozen spec: hotpot_spec.md. Reports positive/mixed/negative honestly."""
import json, re, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

SEED=0; N_EVAL=300; MAXLEN=512; EPS=0.05; TAU0=8.0  # sqrt(d_k), d_k=64
OVERSAMPLE=12  # ~13% retention -> process ~N_EVAL*OVERSAMPLE to reach target
LAYERS=[9,10,11,12]  # 1-indexed upper layers
np.random.seed(SEED); torch.manual_seed(SEED)
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tok=AutoTokenizer.from_pretrained('bert-base-uncased')
m=AutoAdapterModel.from_pretrained('bert-base-uncased', attn_implementation='eager').to(dev).eval()
nm=m.load_adapter('AdapterHub/bert-base-uncased-pf-hotpotqa', source='hf', set_active=True)
m.set_active_adapters(nm)
m.to(dev)  # adapter params load on CPU; re-move whole model to device

# locate encoder layers + value modules for deletion-consequence (block 3)
enc_layers=m.bert.encoder.layer
NHEAD=m.config.num_attention_heads; HDIM=m.config.hidden_size//NHEAD
_vcap={}  # layer_idx(0-based) -> value output [seq, hidden]
def _mk_hook(li0):
    def hk(mod,inp,outp): _vcap[li0]=outp.detach()
    return hk
for L in LAYERS:
    enc_layers[L-1].attention.self.value.register_forward_hook(_mk_hook(L-1))

ds=load_dataset('hotpot_qa','distractor',split='validation')
idx=np.random.RandomState(SEED).permutation(len(ds))[:N_EVAL*OVERSAMPLE]  # oversample; retention filters

def norm(s): return re.sub(r'\s+',' ',s.strip().lower())

def hill(w, q):
    w=np.asarray(w,float); s=w.sum()
    if s<=0: return 0.0
    p=w/s
    if q==np.inf: return 1.0/np.max(p)
    return (np.sum(p**q))**(1.0/(1.0-q))

def build_example(ex):
    """Return tokenization + near/far token masks + gold answer span, or None if excluded."""
    q=ex['question']; ans=ex['answer']
    titles=ex['context']['title']; sents=ex['context']['sentences']  # list[list[str]]
    supp=set(zip(ex['supporting_facts']['title'], ex['supporting_facts']['sent_id']))
    # flatten context; record which char spans are supporting sentences
    ctx=''; sent_spans=[]  # (start,end,is_support)
    for ti,(t,slist) in enumerate(zip(titles,sents)):
        for si,s in enumerate(slist):
            st=len(ctx); ctx+=s; en=len(ctx)
            sent_spans.append((st,en,(t,si) in supp))
    yn = ans.lower() in ('yes','no')
    enc=tok(q, ctx, truncation='only_second', max_length=MAXLEN,
            return_offsets_mapping=True, return_tensors='pt')
    off=enc['offset_mapping'][0].tolist(); ids=enc['input_ids'][0].tolist()
    seqids=enc.sequence_ids(0)
    # locate answer span in context (char) for span answers
    if not yn:
        # search in ORIGINAL ctx (case-insensitive) so char positions match offset_mapping
        pos=ctx.lower().find(ans.lower())
        if pos<0: return None  # answer not in (truncated) context
        aend=pos+len(ans)
        asl=ael=None
        for i,(o,sq) in enumerate(zip(off,seqids)):
            if sq!=1 or o[1]<=o[0]: continue      # skip non-context / special tokens
            if o[0]<=pos<o[1] and asl is None: asl=i
            if o[0]<aend<=o[1]: ael=i
        if asl is None or ael is None or ael<asl: return None
    else:
        asl=ael=None
    # near/far token masks over context tokens
    near=np.zeros(len(ids),bool); far=np.zeros(len(ids),bool)
    supp_present=set()  # which supporting sentences survived truncation
    for i,(o,sq) in enumerate(zip(off,seqids)):
        tokstr=ids[i]
        if sq==0: near[i]=True; continue          # question tokens -> near
        if tokstr in (tok.cls_token_id, tok.sep_token_id): near[i]=True; continue
        if sq!=1: continue                          # padding/None
        # context token: is its char-span inside a supporting sentence?
        c=o[0]; is_sup=False
        for k,(st,en,iss) in enumerate(sent_spans):
            if st<=c<en:
                is_sup=iss
                if iss: supp_present.add(k)
                break
        if is_sup: near[i]=True
        else: far[i]=True
    # also mark [CLS],[SEP] near explicitly
    for i,t in enumerate(ids):
        if t in (tok.cls_token_id, tok.sep_token_id): near[i]=True; far[i]=False
    # retention: ALL supporting sentences present?
    n_supp_total=sum(1 for (_,_,iss) in sent_spans if iss)
    if len(supp_present)<n_supp_total: return None
    if far.sum()==0 or near.sum()==0: return None
    return dict(enc={k:v.to(dev) for k,v in enc.items() if k!='offset_mapping'},
                ids=ids, near=near, far=far, yn=yn, aslot=asl, eslot=ael,
                answer=ans, seqids=seqids, ntok=len(ids))

# ---- main loop ----
rows=[]; kept=0; excl=0; kept_len=[]; excl_len=[]; kept_examples=[]
for ii in idx:
    ex=ds[int(ii)]
    b=build_example(ex)
    if b is None:
        excl+=1; excl_len.append(len(ex['context']['sentences'])); continue
    kept+=1; kept_len.append(b['ntok']); kept_examples.append((int(ii),b))
    with torch.no_grad():
        out=m(**b['enc'], output_attentions=True)
    s_pred=out.start_logits.argmax().item(); e_pred=out.end_logits.argmax().item()
    # decision query positions
    if b['yn']:
        qpos=[b['ids'].index(tok.cls_token_id)]
    else:
        qpos=[s_pred, e_pred]
    near=b['near']; far=b['far']
    # per (layer, head, qpos): logits reconstructed from attention probs? we need pre-softmax logits.
    # attentions are post-softmax probs p_i. Energy E_i = -logit; but we only have probs.
    # Recover relative logits: log p_i = logit_i - logsumexp(logits); so E_i-E_N = -(logit_i-logit_N)
    #  = -(log p_i - log p_N). Use probs directly for m_F; use log-prob gaps for Delta and D_q.
    atts=[out.attentions[L-1][0] for L in LAYERS]  # each: [heads, seq, seq]
    for li,L in enumerate(LAYERS):
        A=atts[li].cpu().numpy()  # [H, seq, seq]
        H=A.shape[0]
        # value vectors this layer: [seq, hidden] -> [H, seq, HDIM]
        V=_vcap[L-1][0].cpu().numpy().reshape(-1,NHEAD,HDIM).transpose(1,0,2)  # [H,seq,HDIM]
        for h in range(H):
            Vh=V[h]  # [seq, HDIM]
            Bmax=float(np.linalg.norm(Vh,axis=1).max())  # max ||v_i|| for Note 10 bound
            for qp in qpos:
                p=A[h,qp,:]  # attention prob over keys (ALREADY at deployed tau0)
                pf=p[far]
                m_F=float(pf.sum())
                # deletion consequence: O=sum p_i v_i ; O_near=renorm over N
                O=p@Vh
                pn=p.copy(); pn[far]=0.0; sn=pn.sum()
                O_near=(pn/sn)@Vh if sn>0 else O*0
                delO=float(np.linalg.norm(O-O_near))
                relpert=delO/(np.linalg.norm(O)+1e-12)
                del_bound=2.0*m_F*Bmax           # Note 10: ||O-O_near|| <= 2 B m_F
                # probs are softmax(l) at tau0, so log p_i = l_i - logZ.  Work in log-prob space:
                # E_i/tau0 = -log p_i (+const); Delta/tau0 = max_N log p - max_F log p (const cancels).
                with np.errstate(divide='ignore'):
                    lp=np.log(np.clip(p,1e-30,None))
                lpN=lp[near].max(); lpF=lp[far].max()
                dtau=float(lpN-lpF)              # Delta/tau0 (dimensionless gap)
                expterm=np.exp(lpF-lpN)          # exp(-Delta/tau0)
                # D_q^F: Hill number is scale-invariant, so shift by max_N cancels -> use far probs
                B2=min(1.0, hill(pf,2)*expterm)
                Binf=min(1.0, hill(pf,np.inf)*expterm)
                Delta=dtau                        # report dimensionless gap
                rows.append(dict(ex=int(ii),L=L,h=h,qp=int(qp),yn=b['yn'],
                    m_F=m_F, Delta=Delta, B2=B2, Binf=Binf,
                    cert2=int(Delta>0 and B2<=EPS), certinf=int(Delta>0 and Binf<=EPS),
                    posgap=int(Delta>0),
                    delO=delO, relpert=relpert, del_bound=del_bound, Bmax=Bmax,
                    correct=int(norm(tok.decode(b['enc']['input_ids'][0][s_pred:e_pred+1]))==norm(b['answer'])) if not b['yn'] else -1))
    if kept>=N_EVAL: break

import pandas as pd
df=pd.DataFrame(rows)
res=dict(
  n_kept=kept, n_excluded=excl,
  retained_frac=kept/(kept+excl),
  kept_len_mean=float(np.mean(kept_len)), 
  n_rows=len(df),
  posgap_frac=float(df.posgap.mean()),
  m_F_mean=float(df.m_F.mean()), m_F_median=float(df.m_F.median()),
  m_F_p90=float(df.m_F.quantile(.9)), m_F_p99=float(df.m_F.quantile(.99)),
  cert2_frac=float(df.cert2.mean()), certinf_frac=float(df.certinf.mean()),
  # zero-violation audit: m_F <= B on positive-gap rows
  viol2=int(((df.Delta>0)&(df.m_F>df.B2+1e-9)).sum()),
  violinf=int(((df.Delta>0)&(df.m_F>df.Binf+1e-9)).sum()),
  n_posgap=int(df.posgap.sum()),
)
# tightness on positive-gap, nonzero-mass rows
pg=df[(df.Delta>0)&(df.m_F>0)]
res['tightness2_median']=float((pg.B2/pg.m_F).median()) if len(pg) else None
# example-level stringent coverage (all rows of an example certified, q=2)
exl=df.groupby('ex').cert2.min()
res['example_cov_stringent']=float(exl.mean())
res['m_F_by_layer']={int(L):float(df[df.L==L].m_F.mean()) for L in LAYERS}
res['cert2_by_layer']={int(L):float(df[df.L==L].cert2.mean()) for L in LAYERS}
# block 3: deletion consequence
res['delO_bound_viol']=int((df.delO>df.del_bound+1e-6).sum())   # Note10 bound must hold
res['relpert_mean']=float(df.relpert.mean()); res['relpert_median']=float(df.relpert.median())
res['relpert_cert_mean']=float(df[df.cert2==1].relpert.mean()) if (df.cert2==1).any() else None
res['relpert_uncert_mean']=float(df[df.cert2==0].relpert.mean())
res['del_tightness_median']=float((df[df.m_F>0].del_bound/(df[df.m_F>0].delO+1e-9)).median())

# ---- block 4: end-to-end masking diagnostic (NON-certified) ----
# Re-run kept examples with far-key attention masked in analysed layers; compare answers.
def f1(pred,gold):
    ps=norm(pred).split(); gs=norm(gold).split()
    if not ps or not gs: return float(ps==gs)
    common={}; 
    for w in ps: common[w]=common.get(w,0)+1
    ov=sum(min(common.get(w,0), gs.count(w)) for w in set(gs))
    if ov==0: return 0.0
    p=ov/len(ps); r=ov/len(gs); return 2*p*r/(p+r)
# analysed-layer-only far masking via forward-pre-hooks on self-attention modules.
# Each hook zeroes the attention WEIGHTS on far key columns and renormalises, but only
# in LAYERS (spec block4: "masked in the analysed layers"), leaving other layers intact.
_e2e_far=[None]      # current example far mask (bool over seq); set per example
_e2e_on=[False]
def _mk_pre(li0):
    self_attn=enc_layers[li0].attention.self
    orig_forward=self_attn.forward
    def patched(hidden_states, attention_mask=None, *a, **kw):
        # build an additive mask that sends far keys to -inf, added to any existing mask
        far=_e2e_far[0]
        if (not _e2e_on[0]) or far is None:
            return orig_forward(hidden_states, attention_mask, *a, **kw)
        seq=hidden_states.shape[1]
        add=torch.zeros(1,1,1,seq, device=hidden_states.device)
        add[0,0,0,torch.tensor(np.where(far)[0],device=hidden_states.device)]=torch.finfo(hidden_states.dtype).min
        am = add if attention_mask is None else attention_mask+add
        return orig_forward(hidden_states, am, *a, **kw)
    return patched
for L in LAYERS:
    sa=enc_layers[L-1].attention.self
    sa.forward=_mk_pre(L-1)

e2e=[]
for ii,b in kept_examples:
    _e2e_on[0]=False; _e2e_far[0]=None
    with torch.no_grad():
        out0=m(**b['enc'], output_attentions=False)
    s0=out0.start_logits.argmax().item(); e0=out0.end_logits.argmax().item()
    pred0=tok.decode(b['enc']['input_ids'][0][s0:e0+1]) if not b['yn'] else b['answer']
    # intervention: mask far keys in analysed layers ONLY
    _e2e_far[0]=b['far']; _e2e_on[0]=True
    with torch.no_grad():
        out1=m(**b['enc'], output_attentions=False)
    _e2e_on[0]=False
    s1=out1.start_logits.argmax().item(); e1=out1.end_logits.argmax().item()
    pred1=tok.decode(b['enc']['input_ids'][0][s1:e1+1]) if not b['yn'] else b['answer']
    e2e.append(dict(em0=int(norm(pred0)==norm(b['answer'])),
                    em1=int(norm(pred1)==norm(b['answer'])),
                    f10=f1(pred0,b['answer']), f11=f1(pred1,b['answer']),
                    agree=int(norm(pred0)==norm(pred1)), yn=b['yn']))
edf=pd.DataFrame(e2e)
res['e2e_n']=len(edf)
res['e2e_EM_before']=float(edf.em0.mean()); res['e2e_EM_after']=float(edf.em1.mean())
res['e2e_F1_before']=float(edf.f10.mean()); res['e2e_F1_after']=float(edf.f11.mean())
res['e2e_agreement']=float(edf.agree.mean())
json.dump(res, open('hotpot_cert.json','w'), indent=2)
df.to_parquet('hotpot_cert_rows.parquet')
print(json.dumps(res,indent=2))
