#!/usr/bin/env python3
"""Normalized (Gibbs) Resolution Calibration certificate on real attention.

The additive RCP bounds the UNNORMALISED far tail. Softmax attention is a NORMALISED
Gibbs field p_i = exp(-E_i/tau)/Z with energies E_i = -<q,k_i>. For a near anchor with
minimal energy E_0 and a far set at energy gap >= Delta (E_i >= E_0 + Delta):
  divide num & denom by exp(-E_0/tau): far weights <= exp(-Delta/tau), Z >= 1, so
  m_F = sum_{far} p_i <= D_q^{(F)} exp(-Delta/tau),        (Gibbs far-mass bound)
where D_q^{(F)} is the Hill number of the far Gibbs weights. Setting RHS = eps:
  tau* = Delta / ln(D_q^{(F)}/eps).                        (certified temperature)
This bounds the NORMALISED far mass directly -- no temperature-dependent source amplitude
V_max(tau), unlike the additive Gaussian view. We validate on real BERT/GPT-2 heads:
 (a) the bound m_F <= D_q exp(-Delta/tau) holds at the deployment tau (0 violations);
 (b) at the certified tau* the measured normalised far mass m_F <= eps.
"""
import json, numpy as np, torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
EPS=0.05; SEED=20260707; QS=[2.0,np.inf]
torch.manual_seed(SEED); np.random.seed(SEED)
dev="cuda" if torch.cuda.is_available() else "cpu"

PARAS=[
"The transformer architecture relies on self-attention to compute representations without recurrence. "
"Each token emits a query, a key, and a value vector, and the output at a position is a weighted average "
"of value vectors, with weights given by a softmax over query-key inner products. Multiple heads run in parallel. ",
"Kernel methods place a similarity function between data points and read out predictions as weighted "
"combinations of stored examples. The bandwidth of the kernel controls locality: a small bandwidth makes "
"the prediction depend only on the closest neighbours, while a large one spreads influence across the set. ",
"A certificate is a guarantee that can be checked without access to the ground truth. In safety-critical "
"deployment one wants to know, before seeing any label, that a model's prediction depends only on nearby "
"training points and cannot be swung by distant or corrupted data. ",
"Neural networks in the infinite-width limit converge to Gaussian processes whose kernel is the neural "
"tangent kernel. Near convergence a finite network behaves approximately as a kernel machine in feature space. ",
]*6

def hill(w,q):
    w=np.clip(w,0,None); s=w.sum()
    if s<=0: return 1.0
    if np.isinf(q): return float(s/(w.max()+1e-30))
    p=w/s; return float((np.power(p,q).sum())**(1.0/(1.0-q)))

def head_gibbs(Q,K,tau_dep):
    # energies E_i = -<q,k_i>; per query define near anchor = min energy, far set by a
    # logit-gap percentile so the near/far split is geometric & label-free.
    G=Q@K.T                                    # [Tq,Tk] inner products (=-E)
    Tq,Tk=G.shape
    res={q:{"bound":[],"meas":[],"viol":0,"taustar":[],"meas_star":[],"star_viol":0} for q in QS}
    n_used=0
    for qi in range(Tq):
        g=G[qi]                                # higher g = nearer (lower energy)
        order=np.argsort(-g)                   # near -> far
        # near anchor group: top decile by logit; far: the rest
        n_near=max(1,int(0.10*Tk)); near_idx=order[:n_near]; far_idx=order[n_near:]
        if len(far_idx)<2: continue
        E0=-g[near_idx].min()                   # min energy over near anchor (= -max logit)
        E_far=-g[far_idx]                        # far energies
        Delta=float(E_far.min()-E0)              # gap: nearest far energy above anchor
        if Delta<=0: continue
        n_used+=1
        # measured normalised far mass at deployment tau
        z=g/tau_dep; z-=z.max(); a=np.exp(z); a/=a.sum()
        m_meas=float(a[far_idx].sum())
        # far Gibbs weights (unnormalised, anchor-shifted) at tau_dep for the Hill count
        wf=np.exp(-(E_far-E0)/tau_dep)
        for q in QS:
            Dq=hill(wf,q)
            bound=float(Dq*np.exp(-Delta/tau_dep))
            res[q]["bound"].append(bound); res[q]["meas"].append(m_meas)
            res[q]["viol"]+= int(m_meas>bound+1e-9)
            # certified tau*: Dq computed at the SAME far set (geometry fixed)
            taustar=Delta/np.log(max(Dq,1.001)/EPS)
            zs=g/taustar; zs-=zs.max(); asx=np.exp(zs); asx/=asx.sum()
            ms=float(asx[far_idx].sum())
            res[q]["taustar"].append(taustar); res[q]["meas_star"].append(ms)
            res[q]["star_viol"]+= int(ms>EPS+1e-6)
    return res,n_used

def run_model(name,is_causal):
    tok=AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    Model=AutoModelForCausalLM if is_causal else AutoModel
    model=Model.from_pretrained(name,attn_implementation="eager").to(dev).eval()
    cfg=model.config
    nL=getattr(cfg,"num_hidden_layers",None) or cfg.n_layer
    nH=getattr(cfg,"num_attention_heads",None) or cfg.n_head
    dm=getattr(cfg,"hidden_size",None) or cfg.n_embd; dh=dm//nH
    tau_dep=float(np.sqrt(dh))                 # deployment temperature = sqrt(d_k)
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
    agg={q:{"bound":[],"meas":[],"viol":0,"nq":0,"taustar":[],"meas_star":[],"star_viol":0} for q in QS}
    Ls=list(range(0,nL,max(1,nL//4)))[:4]; Hs=list(range(0,nH,max(1,nH//3)))[:3]
    longtexts=[" ".join(PARAS[s:s+8]) for s in range(0,len(PARAS),3)]
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
                    res,nu=head_gibbs(Q,K,tau_dep)
                    for q in QS:
                        agg[q]["bound"]+=res[q]["bound"]; agg[q]["meas"]+=res[q]["meas"]
                        agg[q]["viol"]+=res[q]["viol"]; agg[q]["nq"]+=nu
                        agg[q]["taustar"]+=res[q]["taustar"]; agg[q]["meas_star"]+=res[q]["meas_star"]
                        agg[q]["star_viol"]+=res[q]["star_viol"]
    for hd in hs: hd.remove()
    out={"nL":int(nL),"nH":int(nH),"d_head":int(dh),"tau_dep":tau_dep}
    for q in QS:
        b=np.array(agg[q]["bound"]); m=np.array(agg[q]["meas"]); ms=np.array(agg[q]["meas_star"])
        ts=np.array(agg[q]["taustar"])
        key="inf" if np.isinf(q) else int(q)
        out[f"q{key}"]={"n_query_head":int(agg[q]["nq"]),
            "bound_mean":float(b.mean()),"measured_normfar_mean":float(m.mean()),
            "bound_violations":int(agg[q]["viol"]),
            "taustar_mean":float(ts.mean()),"taustar_median":float(np.median(ts)),
            "frac_taustar_below_taudep":float((ts<tau_dep).mean()),
            "taustar_meas_normfar_mean":float(ms.mean()),"taustar_meas_normfar_p99":float(np.percentile(ms,99)),
            "taustar_over_eps_violations":int(agg[q]["star_viol"])}
    return out

results={"eps":EPS,"models":{}}
for name,causal in [("bert-base-uncased",False),("gpt2",True)]:
    try:
        results["models"][name]=run_model(name,causal); print(name,json.dumps(results["models"][name]),flush=True)
    except Exception as e:
        results["models"][name]={"error":repr(e)}; print(name,"FAILED",repr(e),flush=True)
json.dump(results,open("gibbs_attention.json","w"),indent=2)
print("SAVED gibbs_attention.json")
