#!/usr/bin/env python3
"""Attention inherits the sigma* certificate: demonstration on real transformers (long contexts).

Identity: softmax attention weight exp(<q,k_i>/tau) == kernel-field weight
v_i K_sigma(||q-k_i||^2) with v_i=exp(||k_i||^2/2tau), sigma^2=tau (query factor cancels).
So the paper's certificate applies with sigma=sqrt(tau).

We now use LONG contexts (>=256 tokens) so each head has hundreds of keys and the
10th-percentile near/far radius d is a populated, meaningful split (the flagship field
has 20k sources; short sentences with ~15 keys make the quantile degenerate).

Per head we compute, label-free from key geometry:
  d = 10th-pctile query-key distance;  A = V_max*N_eff over far keys at pairwise scale;
  tau* = d^2/(2 ln(A/eps)).
Then measure at tau* and 4*tau*:
  (a) NORMALIZED far attention mass  sum_{far} a_i   (output-relevant: does output see far keys)
  (b) CERTIFIED tail ratio  Tail/(V_max) / eps  where Tail=sum_{far}|v_i|omega_i  (the paper's Prop-1 quantity)
Claim: at tau* the certified tail ratio <= 1 (Prop 1 holds on real attention), and the
normalized far mass is small when the field is populated.
"""
import json, numpy as np, torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

EPS=0.05; SEED=20260707
torch.manual_seed(SEED); np.random.seed(SEED)
dev="cuda" if torch.cuda.is_available() else "cpu"

# long real text (wiki-style paragraphs), concatenated to fill the context window
PARAS = [
"The transformer architecture relies on self-attention to compute representations without recurrence. "
"Each token emits a query, a key, and a value vector, and the output at a position is a weighted average "
"of value vectors, with weights given by a softmax over query-key inner products. Multiple heads run in "
"parallel, each attending to a different learned subspace of the representation. ",
"Kernel methods place a similarity function between data points and read out predictions as weighted "
"combinations of stored examples. The bandwidth of the kernel controls locality: a small bandwidth makes "
"the prediction depend only on the closest neighbours, while a large one spreads influence across the set. "
"Choosing the bandwidth well is classically done by cross-validation against held-out labels. ",
"A certificate is a guarantee that can be checked without access to the ground truth. In safety-critical "
"deployment one wants to know, before seeing any label, that a model's prediction depends only on nearby "
"training points and cannot be swung by distant or corrupted data. Such locality guarantees connect to "
"machine unlearning, influence functions, and the study of memorisation in large models. ",
"Neural networks in the infinite-width limit converge to Gaussian processes whose kernel is the neural "
"tangent kernel. Near convergence a finite network behaves approximately as a kernel machine in this "
"feature space, so questions about locality and interference can be posed in kernel terms even for deep "
"models trained by gradient descent on large corpora of natural language and images. ",
]*6

def pairwise_d2(Q,K):
    return np.maximum(0.0,(Q**2).sum(1)[:,None]+(K**2).sum(1)[None,:]-2*Q@K.T)

def head_certificate(Q,K):
    D2=pairwise_d2(Q,K); R=np.sqrt(D2)
    iu=np.triu_indices(R.shape[0],1)
    d=np.percentile(R[iu],10)
    tau_pair=d*d/(2*np.log(1.0/EPS))
    def measure(tau):
        v=np.exp((K**2).sum(1)/(2*tau)); Vmax=v.max()
        # certified tail (Prop 1) and normalized far mass, per query
        logit=(Q@K.T)/tau; logit-=logit.max(1,keepdims=True)
        a=np.exp(logit); a/=a.sum(1,keepdims=True)
        far=R>d
        tail_ratio=[]; normfar=[]; neffs=[]
        for qi in range(R.shape[0]):
            fm=far[qi]
            w=v[fm]*np.exp(-D2[qi][fm]/(2*tau))     # |v_i| omega_i on far keys (unnorm)
            tail=w.sum()
            tail_ratio.append(tail/Vmax/EPS)         # Prop-1: <=1 iff Tail<=eps*Vmax
            normfar.append(a[qi][fm].sum())
            s1=w.sum(); s2=(w*w).sum(); neffs.append((s1*s1)/s2 if s2>0 else 1.0)
        return np.array(tail_ratio), np.array(normfar), Vmax, np.mean(neffs)
    # A measured at pairwise scale
    _,_,Vmax0,neff0=measure(tau_pair); A=max(Vmax0*neff0,1.01)
    tau_star=d*d/(2*np.log(A/EPS))
    tr_s,nf_s,_,_=measure(tau_star); tr_w,nf_w,_,_=measure(4*tau_star)
    return dict(d=float(d),A=float(A),tau_star=float(tau_star),n_keys=int(K.shape[0]),
                tailratio_star_mean=float(tr_s.mean()),tailratio_star_max=float(tr_s.max()),
                normfar_star_mean=float(nf_s.mean()),normfar_star_max=float(nf_s.max()),
                tailratio_wide_mean=float(tr_w.mean()),normfar_wide_mean=float(nf_w.mean()),
                tail_viol=int((tr_s>1.0).sum()),n_q=int(len(tr_s)))

results={"eps":EPS,"identity_max_diff":None,"models":{}}
rng=np.random.default_rng(SEED)
qd,kd=rng.normal(size=32),rng.normal(size=(64,32))*rng.uniform(.5,2,(64,1))
tau=0.7; lg=kd@qd/tau; a1=np.exp(lg-lg.max()); a1/=a1.sum()
d2=((qd[None,:]-kd)**2).sum(1); vv=np.exp((kd**2).sum(1)/(2*tau)); wf=vv*np.exp(-d2/(2*tau)); a2=wf/wf.sum()
results["identity_max_diff"]=float(np.abs(a1-a2).max())

def run_model(name,is_causal):
    tok=AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    Model=AutoModelForCausalLM if is_causal else AutoModel
    model=Model.from_pretrained(name,attn_implementation="eager").to(dev).eval()
    cfg=model.config
    nL=getattr(cfg,"num_hidden_layers",None) or cfg.n_layer
    nH=getattr(cfg,"num_attention_heads",None) or cfg.n_head
    dm=getattr(cfg,"hidden_size",None) or cfg.n_embd; dh=dm//nH
    maxlen=min(getattr(cfg,"max_position_embeddings",512) or 512,384)
    cap={}
    base=model if not is_causal else model.transformer
    layers=base.encoder.layer if hasattr(base,"encoder") else base.h
    def mk(li):
        def hook(m,i,o): cap[li]=o.detach()
        return hook
    hs=[]
    for li,lyr in enumerate(layers):
        if is_causal: hs.append(lyr.attn.c_attn.register_forward_hook(mk(li)))
        else:
            hs.append(lyr.attention.self.query.register_forward_hook(mk(("q",li))))
            hs.append(lyr.attention.self.key.register_forward_hook(mk(("k",li))))
    rows=[]; real_id_diffs=[]
    Ls=list(range(0,nL,max(1,nL//4)))[:4]; Hs=list(range(0,nH,max(1,nH//3)))[:3]
    # build a few LONG sequences
    longtexts=[]
    for start in range(0,len(PARAS),3):
        longtexts.append(" ".join(PARAS[start:start+8]))
    with torch.no_grad():
        for s in longtexts[:8]:
            enc=tok(s,return_tensors="pt",truncation=True,max_length=maxlen).to(dev)
            T=enc["input_ids"].shape[1]
            if T<128: continue
            cap.clear(); model(**enc)
            for li in Ls:
                if is_causal:
                    qkv=cap[li][0]; Qf,Kf=qkv[:,:dm],qkv[:,dm:2*dm]
                else:
                    Qf,Kf=cap[("q",li)][0],cap[("k",li)][0]
                Qf=Qf.reshape(T,nH,dh); Kf=Kf.reshape(T,nH,dh)
                for h in Hs:
                    Q=Qf[:,h,:].float().cpu().numpy(); K=Kf[:,h,:].float().cpu().numpy()
                    # REAL-QK identity check (log-space, stable): softmax over attention
                    # logits == softmax over kernel-field logits (log v_i - ||q-k||^2/2tau).
                    tau_id=1.0
                    lg=(Q@K.T)/tau_id                                  # attention logits
                    fl=((K**2).sum(1)[None,:] - pairwise_d2(Q,K))/(2*tau_id)  # field logits = (||k||^2-||q-k||^2)/2tau
                    lg=lg-lg.max(1,keepdims=True); fl=fl-fl.max(1,keepdims=True)
                    aa=np.exp(lg); aa/=aa.sum(1,keepdims=True)
                    af=np.exp(fl); af/=af.sum(1,keepdims=True)
                    real_id_diffs.append(float(np.abs(aa-af).max()))
                    rows.append(head_certificate(Q,K))
    for hd in hs: hd.remove()
    tr=np.array([r["tailratio_star_mean"] for r in rows]); trmax=np.array([r["tailratio_star_max"] for r in rows])
    nf=np.array([r["normfar_star_mean"] for r in rows])
    return dict(nL=int(nL),nH=int(nH),d_head=int(dh),n_head_samples=len(rows),
                real_qk_identity_max_diff=float(max(real_id_diffs)) if real_id_diffs else None,
                mean_keys=float(np.mean([r["n_keys"] for r in rows])),
                tailratio_star_mean=float(tr.mean()),tailratio_star_p95=float(np.percentile(tr,95)),
                tailratio_star_headmax=float(trmax.max()),
                tailratio_wide_mean=float(np.mean([r["tailratio_wide_mean"] for r in rows])),
                normfar_star_mean=float(nf.mean()),normfar_star_p95=float(np.percentile(nf,95)),
                normfar_wide_mean=float(np.mean([r["normfar_wide_mean"] for r in rows])),
                tail_violations=int(sum(r["tail_viol"] for r in rows)),total_q=int(sum(r["n_q"] for r in rows)),
                frac_heads_tailratio_below1=float((tr<=1.0).mean()))
for name,causal in [("bert-base-uncased",False),("gpt2",True)]:
    try:
        results["models"][name]=run_model(name,causal); print(name,"done:",json.dumps(results["models"][name]))
    except Exception as e:
        results["models"][name]={"error":repr(e)}; print(name,"FAILED:",repr(e))
json.dump(results,open("attention_cert.json","w"),indent=2)
print("IDENTITY max diff:",results["identity_max_diff"]); print("SAVED")
