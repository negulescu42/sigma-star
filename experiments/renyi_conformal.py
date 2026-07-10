#!/usr/bin/env python3
"""RCP upgrades on the three frozen CIFAR backbones:
 (1) Renyi/Hill D_q family: participation ratio N_eff is the q=2 Hill number; generalise
     to D_q (q in {2,3,5,inf}). Bound sum omega <= D_q K_sigma(d); D_inf<=D_q<=D_2 so larger q
     is an equal-or-tighter certificate. Certified sigma*_q = d/sqrt(2 ln(D_q/eps)) (V_max=1).
     Report tail/eps and accuracy at each q, and monotonicity of D_q(sigma) (should be nondecreasing).
 (2) Conformal finite-sample query certificate: from n unlabeled calibration queries take the
     order statistic A_conf=A_(k), k=ceil((n+1)(1-alpha)); exchangeability => a future query has
     A_y<=A_conf w.p.>=1-alpha. Validate empirically on held-out queries.
 (3) Underflow audit + 1-NN baseline: recompute the small-sigma Parzen classifier with a
     numerically stable log-domain readout; confirm near-delta sigma -> 1-NN (not chance).
seed=0 to match p99cert.py / extra_baselines_a2.py.
"""
import numpy as np, json, time, sys, torch
t0=time.time(); dev="cuda" if torch.cuda.is_available() else "cpu"
EPS=0.05; ALPHA=0.05; rng=np.random.default_rng(0)
FEATS=[("/workspace/cifar_feats.npz","resnet18"),
       ("/workspace/cifar_feats_r50.npz","resnet50"),
       ("/workspace/cifar_feats_vit.npz","vit")]
Ksig=lambda s,r2: torch.exp(-r2/(2*s*s))
QS=[2.0,3.0,5.0,np.inf]

def hill_Dq(w, q):
    # w: [B, M] nonneg far weights (0 where not-far); D_q of each row over its nonzero support
    s1=w.sum(1)
    if np.isinf(q):
        mx=w.max(1); return s1/(mx+1e-30)
    sq=(w.clip(min=0)**q).sum(1)
    p_sum = (w/ (s1[:,None]+1e-30))
    Dq = ((p_sum**q).sum(1))**(1.0/(1.0-q))
    return Dq

def run(feat,name):
    z=np.load(feat); Xtr,ytr,Xte,yte=z["Xtr"],z["Ytr"],z["Xte"],z["Yte"]
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    dim=Xtr.shape[1]; ncls=int(max(ytr.max(),yte.max())+1)
    keep=np.concatenate([rng.choice(np.where(ytr==c)[0],200,replace=False) for c in range(ncls)])
    Xk=torch.tensor(Xtr[keep],device=dev,dtype=torch.float32); yk=torch.tensor(ytr[keep],device=dev)
    oh=torch.zeros(len(yk),ncls,device=dev); oh[torch.arange(len(yk)),yk]=1.0
    Xte_t=torch.tensor(Xte,device=dev,dtype=torch.float32); yte_t=torch.tensor(yte,device=dev)
    n_k=len(Xk); n_te=Xte_t.size(0)
    sub=Xk[rng.choice(n_k,4000,replace=False)]
    pd=torch.cdist(sub,sub); d=torch.quantile(pd[pd>0][:2_000_000],0.10).item(); d2=d*d
    sig_pair=d/np.sqrt(2*np.log(1/EPS))

    @torch.no_grad()
    def far_weights(sig,q_idx,nq=10000):
        nq=min(nq,n_te); q=Xte_t[rng.choice(n_te,nq,replace=False)]; W=[]
        for i in range(0,nq,500):
            D2=torch.cdist(q[i:i+500],Xk).pow(2); fm=(D2>d2).float()
            W.append((Ksig(sig,D2)*fm).cpu().numpy())
        return np.concatenate(W)  # [nq, n_k]
    @torch.no_grad()
    def tail_over_eps(sig,nq=10000):
        nq=min(nq,n_te); q=Xte_t[rng.choice(n_te,nq,replace=False)]; out=[]
        for i in range(0,nq,500):
            D2=torch.cdist(q[i:i+500],Xk).pow(2); fm=(D2>d2).float()
            out.append((Ksig(sig,D2)*fm).sum(1).cpu().numpy())
        return np.concatenate(out)/EPS
    @torch.no_grad()
    def acc_stable(sig,bs=2000):
        # log-domain Parzen: class score = logsumexp_{i in c}(-D2/2sig^2); argmax. Underflow-proof.
        c=0
        for i in range(0,n_te,bs):
            D2=torch.cdist(Xte_t[i:i+bs],Xk).pow(2); logw=-D2/(2*sig*sig)
            # scatter-logsumexp over classes
            m=logw.max(1,keepdim=True).values; ew=torch.exp(logw-m)
            cls=(ew@oh); score=torch.log(cls+1e-30)+m
            c+=(score.argmax(1)==yte_t[i:i+bs]).sum().item()
        return c/n_te
    @torch.no_grad()
    def acc_naive(sig,bs=2000):
        # the ORIGINAL (unstable) readout: linear-domain weighted vote, underflows at small sigma
        c=0
        for i in range(0,n_te,bs):
            D2=torch.cdist(Xte_t[i:i+bs],Xk).pow(2); pr=(Ksig(sig,D2)@oh).argmax(1)
            c+=(pr==yte_t[i:i+bs]).sum().item()
        return c/n_te
    @torch.no_grad()
    def acc_1nn(bs=2000):
        c=0
        for i in range(0,n_te,bs):
            D2=torch.cdist(Xte_t[i:i+bs],Xk).pow(2); nn=D2.argmin(1)
            c+=(yk[nn]==yte_t[i:i+bs]).sum().item()
        return c/n_te

    # ---------- (1) Renyi D_q family ----------
    W=far_weights(sig_pair,0)     # far weights at pairwise scale [nq, n_k]
    dq_res={}
    for q in QS:
        Dq=hill_Dq(W,q); A_q=float(np.mean(Dq))   # V_max=1
        sig_q=d/np.sqrt(2*np.log(max(A_q,1.001)/EPS))
        te=tail_over_eps(sig_q); ac=acc_stable(sig_q)
        dq_res[("inf" if np.isinf(q) else int(q))]={
            "A_q":A_q,"sigma_q":float(sig_q),"tail_mean":float(te.mean()),
            "tail_p99":float(np.percentile(te,99)),"tail_max":float(te.max()),
            "frac_over_1":float((te>1).mean()),"acc":ac}
    # monotonicity of D_q(sigma): grid of sigma, mean D_q nondecreasing?
    sig_grid=np.linspace(0.5*sig_pair,3*sig_pair,10)
    mono={}
    for q in QS:
        means=[]
        for s in sig_grid:
            Ws=far_weights(s,0,nq=3000); means.append(float(np.mean(hill_Dq(Ws,q))))
        diffs=np.diff(means); mono[("inf" if np.isinf(q) else int(q))]={
            "means":[float(x) for x in means],"violations":int((diffs< -1e-6).sum())}

    # ---------- (2) conformal query certificate ----------
    Wc=far_weights(sig_pair,0,nq=n_te)
    A_all=hill_Dq(Wc,2.0)   # q=2 (participation ratio) per-query mass
    ncal=int(0.7*len(A_all)); cal=A_all[:ncal]; val=A_all[ncal:]
    k=int(np.ceil((ncal+1)*(1-ALPHA))); k=min(k,ncal)
    A_conf=float(np.sort(cal)[k-1])
    cov=float((val<=A_conf).mean())                       # should be >= 1-alpha
    sig_conf=d/np.sqrt(2*np.log(A_conf/EPS))
    sig_mean=d/np.sqrt(2*np.log(float(cal.mean())/EPS))
    # per-query tail at sig_conf on validation: fraction within budget
    te_conf=tail_over_eps(sig_conf)
    conf={"alpha":ALPHA,"n_cal":ncal,"n_val":int(len(val)),"k":k,
          "A_mean":float(cal.mean()),"A_conf":A_conf,
          "coverage_A_le_Aconf":cov,"sigma_mean":float(sig_mean),"sigma_conf":float(sig_conf),
          "tail_conf_mean":float(te_conf.mean()),"tail_conf_p99":float(np.percentile(te_conf,99)),
          "tail_conf_frac_over_1":float((te_conf>1).mean())}

    # ---------- (3) underflow audit + 1-NN ----------
    sig_sj=0.05*d   # near-delta scale in the SJ regime
    under={"sigma_near_delta":float(sig_sj),
           "acc_naive_lineardomain":acc_naive(sig_sj),
           "acc_stable_logdomain":acc_stable(sig_sj),
           "acc_1nn":acc_1nn(),
           "acc_naive_at_sigstar":acc_naive(dq_res[2]["sigma_q"]),
           "acc_stable_at_sigstar":acc_stable(dq_res[2]["sigma_q"])}
    r={"name":name,"dim":dim,"eps":EPS,"d":float(d),"sigma_pair":float(sig_pair),
       "renyi":dq_res,"monotonicity":mono,"conformal":conf,"underflow":under}
    print(f"[{name}] D2 sig={dq_res[2]['sigma_q']:.3f} acc={dq_res[2]['acc']:.4f} | "
          f"Dinf sig={dq_res['inf']['sigma_q']:.3f} acc={dq_res['inf']['acc']:.4f} tail={dq_res['inf']['tail_mean']:.3f} | "
          f"conf cov={cov:.4f}(>= {1-ALPHA}) | underflow naive={under['acc_naive_lineardomain']:.4f} "
          f"stable={under['acc_stable_logdomain']:.4f} 1nn={under['acc_1nn']:.4f}",flush=True)
    return r

OUT={"eps":EPS,"alpha":ALPHA,"fields":[]}
for feat,name in FEATS:
    OUT["fields"].append(run(feat,name))
json.dump(OUT,open("renyi_conformal.json","w"),indent=2)
print(f"WROTE renyi_conformal.json {time.time()-t0:.1f}s")
