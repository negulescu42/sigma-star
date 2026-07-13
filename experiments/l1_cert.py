
"""WP1.4 + WP5.1: l1 vector decision certificate recompute + hard-truncation comparator.
Frozen ResNet-18/CIFAR-100, same setup as perf_cert_verify.py.
Reports certified-stable fraction under BOTH the old l_inf (margin>2*tail) and the
sharper l1 deployment threshold (margin>tail), the guarantee-holds rate for each,
and a hard-truncated-Gaussian comparator (exact locality via compact support).
"""
import numpy as np, torch, json
def log(*a): print(*a,flush=True)
DEV="cuda"; rng=np.random.default_rng(0); EPS=0.05
z=np.load("/workspace/cifar_feats.npz"); Xtr,Ytr,Xte,Yte=z["Xtr"],z["Ytr"],z["Xte"],z["Yte"]
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
keep=np.concatenate([rng.choice(np.where(Ytr==c)[0],200,replace=False) for c in range(100)])
Xk,Yk=Xtr[keep],Ytr[keep]
Xk_t=torch.tensor(Xk,device=DEV,dtype=torch.float32); Yk_t=torch.tensor(Yk,device=DEV)
oh=torch.zeros(len(Yk),100,device=DEV); oh[torch.arange(len(Yk)),Yk_t]=1.0
Xte_t=torch.tensor(Xte,device=DEV,dtype=torch.float32); Yte_t=torch.tensor(Yte,device=DEV)
sub=Xk_t[rng.choice(len(Xk),4000,replace=False)]
pd=torch.cdist(sub,sub); d=torch.quantile(pd[pd>0][:2_000_000],0.10).item()
def Ksig(s,r): return torch.exp(-r/(2*s*s))
sigma_pair=d/np.sqrt(2*np.log(1/EPS))
@torch.no_grad()
def Aref(sref,nq=2000):
    q=Xte_t[rng.choice(Xte_t.size(0),nq,replace=False)]; v=[]
    for i in range(0,nq,500):
        D2=torch.cdist(q[i:i+500],Xk_t).pow(2); fm=D2>d*d
        w=Ksig(sref,D2)*fm; s=w.sum(1); s2=(w*w).sum(1); v.append(((s*s)/(s2+1e-30)).cpu().numpy())
    return float(np.concatenate(v).mean())
A=Aref(sigma_pair); sigma_star=d/np.sqrt(2*np.log(A/EPS))
grid=np.geomspace(0.5,30,30); cvidx=rng.choice(len(Xtr),5000,replace=False)
Xv=torch.tensor(Xtr[cvidx],device=DEV,dtype=torch.float32); Yv=torch.tensor(Ytr[cvidx],device=DEV)
@torch.no_grad()
def acc_on(Xq,Yq,s,bs=2000):
    inv=1/(2*s*s); c=0
    for i in range(0,Xq.size(0),bs):
        d2=torch.cdist(Xq[i:i+bs],Xk_t).pow(2); c+=((torch.exp(-d2*inv)@oh).argmax(1)==Yq[i:i+bs]).sum().item()
    return c/Xq.size(0)
s_cv=grid[int(np.argmax([acc_on(Xv,Yv,s) for s in grid]))]
log(f"d={d:.3f} sigma_pair={sigma_pair:.3f} A_ref={A:.1f} sigma_star={sigma_star:.3f} sigma_cv={s_cv:.3f}")

@torch.no_grad()
def certify(sig,bs=1000):
    inv=1/(2*sig*sig); N=Xte_t.size(0)
    dep_correct=loc_correct=dep_eq_loc=0
    stable_linf=stable_l1=0                 # margin>2tail (old) vs margin>tail (l1 deploy)
    seq_linf=seq_l1=0                       # of those stable, dep==loc (guarantee holds)
    far_max_all=0.0; tail_all=0.0
    for i in range(0,N,bs):
        xb=Xte_t[i:i+bs]; yb=Yte_t[i:i+bs]
        D2=torch.cdist(xb,Xk_t).pow(2); K=torch.exp(-D2*inv)
        near=(D2<=d*d).float(); far=1.0-near
        Sn=(K*near)@oh; Sf=(K*far)@oh; S=Sn+Sf
        tail=(K*far).sum(1)                 # ||S_far||_1  (Vmax=1, one-hot => l1 mass = total far weight)
        far_max=Sf.max(1).values
        dep=S.argmax(1); loc=Sn.argmax(1)
        top2=Sn.topk(2,dim=1).values; locmargin=top2[:,0]-top2[:,1]
        st_linf=(locmargin>2*tail); st_l1=(locmargin>tail)
        dep_correct+=(dep==yb).sum().item(); loc_correct+=(loc==yb).sum().item()
        dep_eq_loc+=(dep==loc).sum().item()
        stable_linf+=st_linf.sum().item(); stable_l1+=st_l1.sum().item()
        seq_linf+=(st_linf&(dep==loc)).sum().item(); seq_l1+=(st_l1&(dep==loc)).sum().item()
        far_max_all+=far_max.sum().item(); tail_all+=tail.sum().item()
    n=N
    return dict(sig=float(sig), dep_acc=dep_correct/n, loc_acc=loc_correct/n, dep_eq_loc=dep_eq_loc/n,
                stable_linf=stable_linf/n, stable_l1=stable_l1/n,
                guarantee_linf=seq_linf/max(stable_linf,1), guarantee_l1=seq_l1/max(stable_l1,1),
                mean_tail_over_eps=(tail_all/n)/EPS, mean_farmax_over_eps=(far_max_all/n)/EPS)
out={}
for name,sig in [("pair",sigma_pair),("star",sigma_star),("cv",s_cv)]:
    out[name]=certify(sig); r=out[name]
    log(f"[{name} {sig:.2f}] dep_acc={r['dep_acc']:.3f} loc_acc={r['loc_acc']:.3f} "
        f"stable_linf(2t)={r['stable_linf']:.3f} stable_l1(t)={r['stable_l1']:.3f} "
        f"guar_linf={r['guarantee_linf']:.4f} guar_l1={r['guarantee_l1']:.4f} tail/eps={r['mean_tail_over_eps']:.2f}")

# WP5.1 hard-truncated Gaussian: K*1{r<=d}. Bandwidth by same oracle grid.
@torch.no_grad()
def acc_trunc(Xq,Yq,s,bs=2000):
    inv=1/(2*s*s); c=0
    for i in range(0,Xq.size(0),bs):
        d2=torch.cdist(Xq[i:i+bs],Xk_t).pow(2); K=torch.exp(-d2*inv)*(d2<=d*d).float()
        c+=((K@oh).argmax(1)==Yq[i:i+bs]).sum().item()
    return c/Xq.size(0)
s_cv_trunc=grid[int(np.argmax([acc_trunc(Xv,Yv,s) for s in grid]))]
@torch.no_grad()
def eval_trunc(sig,bs=1000):
    inv=1/(2*sig*sig); N=Xte_t.size(0); cor=0; agree=0
    # agreement with smooth full field at sigma_star
    invs=1/(2*sigma_star*sigma_star)
    for i in range(0,N,bs):
        xb=Xte_t[i:i+bs]; yb=Yte_t[i:i+bs]; d2=torch.cdist(xb,Xk_t).pow(2)
        Kt=torch.exp(-d2*inv)*(d2<=d*d).float(); pt=(Kt@oh).argmax(1)
        Ks=torch.exp(-d2*invs); ps=(Ks@oh).argmax(1)
        cor+=(pt==yb).sum().item(); agree+=(pt==ps).sum().item()
    return dict(sig=float(sig), acc=cor/N, agree_with_smooth_star=agree/N, far_mass=0.0)
out["trunc"]=eval_trunc(s_cv_trunc)
log(f"[trunc oracle sig={s_cv_trunc:.2f}] acc={out['trunc']['acc']:.3f} agree_w_star={out['trunc']['agree_with_smooth_star']:.3f} far_mass=0")
out["meta"]=dict(d=d,sigma_pair=float(sigma_pair),A_ref=A,sigma_star=float(sigma_star),
                 sigma_cv=float(s_cv),sigma_cv_trunc=float(s_cv_trunc),eps=EPS)
json.dump(out,open("l1_cert.json","w"),indent=2); log("DONE")
