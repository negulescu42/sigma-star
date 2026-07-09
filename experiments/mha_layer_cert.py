#!/usr/bin/env python3
"""Multi-head layer propagation of the sigma* certificate on a real BERT layer.

Single-head result (paper Section 4 / SI Note 9): at the certified temperature tau*,
softmax attention over keys is a Gaussian kernel field and the far-key interference tail
is <= eps (in V_max units). Per head the OUTPUT-relevant quantity is the normalized far
mass m_h = sum_{far} a_i^h.

This script propagates that per-head quantity through value mixing and the output
projection to bound the FULL attention-layer output perturbation under deletion of the
far set. For query position q, head output o^h = sum_i a_i^h val_i^h, val_i^h = W_V^h x_i.
Deleting the far keys (renormalizing softmax over the near keys) moves o^h by at most
  ||Delta o^h|| <= 2 m_h max_i ||val_i^h||   (triangle ineq + renormalization; see SI proof)
and max_i ||val_i^h|| <= ||W_V^h||_2 X with X = max_i ||x_i||. The layer output is
O = W_O concat_h(o^h), so
  ||Delta O||_2 <= ||W_O||_2 sqrt( sum_h ( 2 m_h ||W_V^h||_2 X )^2 ).
This bound is computable from pretrained weights + the per-head far masses. We VALIDATE it
by measuring the actual ||Delta O|| from literally deleting the far keys and recomputing
the layer output, confirming actual <= bound and reporting the looseness factor.

Honest scope: the per-head far mass m_h at tau* is 0.35-0.62 for attention (unbounded key
norms), NOT eps-small, so the layer bound is a bounded, computable perturbation that is
tight in the bounded-weight regime and loose here. This is the single-head scoping
propagated to the layer, not an eps-locality claim for transformer layers.
"""
import json, numpy as np, torch
from transformers import AutoModel, AutoTokenizer

EPS=0.05; SEED=20260707
torch.manual_seed(SEED); np.random.seed(SEED)
dev="cuda" if torch.cuda.is_available() else "cpu"

PARAS = [
"The transformer architecture relies on self-attention to compute representations without recurrence. "
"Each token emits a query, a key, and a value vector, and the output at a position is a weighted average "
"of value vectors, with weights given by a softmax over query-key inner products. Multiple heads run in "
"parallel, each attending to a different learned subspace of the representation. ",
"Kernel methods place a similarity function between data points and read out predictions as weighted "
"combinations of stored examples. The bandwidth of the kernel controls locality: a small bandwidth makes "
"the prediction depend only on the closest neighbours, while a large one spreads influence across the set. ",
"A certificate is a guarantee that can be checked without access to the ground truth. In safety-critical "
"deployment one wants to know, before seeing any label, that a model's prediction depends only on nearby "
"training points and cannot be swung by distant or corrupted data. ",
"Neural networks in the infinite-width limit converge to Gaussian processes whose kernel is the neural "
"tangent kernel. Near convergence a finite network behaves approximately as a kernel machine in this "
"feature space, so questions about locality and interference can be posed in kernel terms. ",
]*6

def spectral_norm(W):  # W: numpy 2D
    return float(np.linalg.svd(W, compute_uv=False)[0])

def run():
    name="bert-base-uncased"
    tok=AutoTokenizer.from_pretrained(name)
    model=AutoModel.from_pretrained(name, attn_implementation="eager").to(dev).eval()
    cfg=model.config
    nL,nH,dm=cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
    dh=dm//nH; maxlen=min(cfg.max_position_embeddings or 512, 384)
    layers=model.encoder.layer

    # build long contexts
    longtexts=[" ".join(PARAS[s:s+8]) for s in range(0,len(PARAS),3)]
    Ls=list(range(0,nL,max(1,nL//4)))[:4]   # sample 4 layers
    results={"eps":EPS,"model":name,"nL":nL,"nH":nH,"d_head":dh,"layers":{}}

    for li in Ls:
        lyr=layers[li]
        # weights: value.weight (dm x dm), output.dense.weight (dm x dm)
        Wv=lyr.attention.self.value.weight.detach().float().cpu().numpy()   # (dm, dm) rows=out
        bv=lyr.attention.self.value.bias.detach().float().cpu().numpy()
        Wo=lyr.attention.output.dense.weight.detach().float().cpu().numpy() # (dm, dm)
        Wo_norm=spectral_norm(Wo)
        # per-head value weight slice: output dims [h*dh:(h+1)*dh], all input dims
        Wv_head_norm=[spectral_norm(Wv[h*dh:(h+1)*dh,:]) for h in range(nH)]

        # capture hidden states INTO this attention layer (input to Q/K/V)
        cap={}
        def hook(m,i,o): cap["hs"]=i[0].detach()
        hh=lyr.attention.self.query.register_forward_hook(hook)
        per_ctx=[]
        with torch.no_grad():
            for s in longtexts[:6]:
                enc=tok(s,return_tensors="pt",truncation=True,max_length=maxlen).to(dev)
                T=enc["input_ids"].shape[1]
                if T<128: continue
                cap.clear(); model(**enc)
                X=cap["hs"][0].float().cpu().numpy()            # (T, dm) input hidden states
                Xnorm=float(np.max(np.linalg.norm(X,axis=1)))   # max ||x_i||
                # Q,K,V per head
                Q=(X@lyr.attention.self.query.weight.detach().float().cpu().numpy().T
                   + lyr.attention.self.query.bias.detach().float().cpu().numpy()).reshape(T,nH,dh)
                K=(X@lyr.attention.self.key.weight.detach().float().cpu().numpy().T
                   + lyr.attention.self.key.bias.detach().float().cpu().numpy()).reshape(T,nH,dh)
                V=(X@Wv.T + bv).reshape(T,nH,dh)                # (T,nH,dh) value vectors

                # per head: tau* from key geometry (label-free, as in attention_cert)
                mh=np.zeros((T,nH)); valmax=np.zeros(nH)
                dOh=np.zeros((T,nH,dh))  # actual per-head output change under far-deletion
                for h in range(nH):
                    Kh=K[:,h,:]; Qh=Q[:,h,:]; Vh=V[:,h,:]
                    valmax[h]=float(np.max(np.linalg.norm(Vh,axis=1)))
                    D2=np.maximum(0.0,(Qh**2).sum(1)[:,None]+(Kh**2).sum(1)[None,:]-2*Qh@Kh.T)
                    R=np.sqrt(D2); iu=np.triu_indices(T,1); d=np.percentile(R[iu],10)
                    # A at pairwise scale, then tau*
                    tau_pair=d*d/(2*np.log(1.0/EPS))
                    def neff_at(tau):
                        w=np.exp((Kh**2).sum(1)/(2*tau)); Vmax=w.max()
                        far=R>d; ne=[]
                        for qi in range(T):
                            fm=far[qi]; ww=w[fm]*np.exp(-D2[qi][fm]/(2*tau))
                            s1=ww.sum(); s2=(ww*ww).sum(); ne.append((s1*s1)/s2 if s2>0 else 1.0)
                        return Vmax, np.mean(ne)
                    Vmax0,ne0=neff_at(tau_pair); A=max(Vmax0*ne0,1.01)
                    tau_star=d*d/(2*np.log(A/EPS))
                    # softmax at tau*, normalized far mass, and actual far-deletion perturbation
                    lg=(Qh@Kh.T)/tau_star; lg-=lg.max(1,keepdims=True)
                    a=np.exp(lg); a/=a.sum(1,keepdims=True); far=R>d
                    o_all=a@Vh                                   # (T,dh)
                    a_near=a.copy(); a_near[far]=0.0
                    row=a_near.sum(1,keepdims=True); row[row==0]=1e-12
                    a_near/=row
                    o_near=a_near@Vh
                    dOh[:,h,:]=o_all-o_near
                    mh[:,h]=a[far.any(1)].sum(1) if False else (a*far).sum(1)
                # layer output change: O = W_O concat_h(o^h); DeltaO = W_O concat(Delta o^h)
                dO_concat=dOh.reshape(T,nH*dh)                   # (T, dm)
                dO=dO_concat@Wo.T                                # (T, dm) actual layer-output change
                dO_norm=np.linalg.norm(dO,axis=1)                # per query
                # bound per query: ||W_O|| sqrt(sum_h (2 m_h ||W_V^h|| X)^2)
                bnd=np.zeros(T)
                for qi in range(T):
                    terms=np.array([2*mh[qi,h]*Wv_head_norm[h]*Xnorm for h in range(nH)])
                    bnd[qi]=Wo_norm*np.sqrt((terms**2).sum())
                # also a tighter measured-val variant: replace ||W_V^h|| X by measured max||val||
                bnd_meas=np.zeros(T)
                for qi in range(T):
                    terms=np.array([2*mh[qi,h]*valmax[h] for h in range(nH)])
                    bnd_meas[qi]=Wo_norm*np.sqrt((terms**2).sum())
                O_all_norm=float(np.median(np.linalg.norm((V.reshape(T,nH*dh))@Wo.T,axis=1)))
                per_ctx.append(dict(T=int(T),
                    mean_far_mass=float(mh.mean()), max_far_mass=float(mh.max()),
                    actual_dO_mean=float(dO_norm.mean()), actual_dO_max=float(dO_norm.max()),
                    bound_mean=float(bnd.mean()), bound_max=float(bnd.max()),
                    bound_meas_mean=float(bnd_meas.mean()),
                    holds=bool((dO_norm<=bnd+1e-6).all()),
                    holds_meas=bool((dO_norm<=bnd_meas+1e-6).all()),
                    looseness_median=float(np.median(bnd/np.maximum(dO_norm,1e-9))),
                    looseness_meas_median=float(np.median(bnd_meas/np.maximum(dO_norm,1e-9))),
                    Wo_norm=Wo_norm, Wv_head_norm_mean=float(np.mean(Wv_head_norm)), Xnorm=Xnorm,
                    O_typ_norm=O_all_norm,
                    rel_pert_median=float(np.median(dO_norm)/max(O_all_norm,1e-9))))
        hh.remove()
        # aggregate over contexts
        agg=lambda k: float(np.mean([c[k] for c in per_ctx]))
        results["layers"][str(li)]=dict(
            n_ctx=len(per_ctx),
            mean_far_mass=agg("mean_far_mass"), max_far_mass=float(max(c["max_far_mass"] for c in per_ctx)),
            actual_dO_mean=agg("actual_dO_mean"), actual_dO_max=float(max(c["actual_dO_max"] for c in per_ctx)),
            bound_mean=agg("bound_mean"),
            bound_holds_all=all(c["holds"] for c in per_ctx),
            bound_meas_holds_all=all(c["holds_meas"] for c in per_ctx),
            looseness_median=agg("looseness_median"),
            looseness_meas_median=agg("looseness_meas_median"),
            Wo_norm=agg("Wo_norm"), Wv_head_norm_mean=agg("Wv_head_norm_mean"),
            rel_pert_median=agg("rel_pert_median"))
        print("layer",li,json.dumps(results["layers"][str(li)]))
    json.dump(results, open("mha_layer_cert.json","w"), indent=2)
    print("SAVED"); return results

run()
