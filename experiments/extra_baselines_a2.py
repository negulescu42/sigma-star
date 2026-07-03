"""Referee follow-up: (1) Sheather-Jones + GP marginal-likelihood baselines on the
three frozen CIFAR backbones, computed with the SAME evaluation as the main tables;
(2) a label-free A2 (near/far separation) diagnostic across the CIFAR fields plus
sklearn public datasets, predicting where sigma* matches labelled grid search vs trails it.
seed=0 to match p99cert.py."""
import numpy as np, json, time, sys, torch
t0=time.time(); dev="cuda" if torch.cuda.is_available() else "cpu"; EPS=0.05
rng=np.random.default_rng(0)
Ksig=lambda s,r2: torch.exp(-r2/(2*s*s))

def prep(Xtr,ytr,Xte,yte):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
    return (Xtr-mu)/sd,(Xte-mu)/sd

def evaluate(Xtr,ytr,Xte,yte,name,dim,nq=6000,ncls=None):
    Xtr,Xte=prep(Xtr,ytr,Xte,yte)
    ncls=int(max(ytr.max(),yte.max())+1) if ncls is None else ncls
    # keep set: up to 200/class
    idx=[]
    for c in range(ncls):
        ci=np.where(ytr==c)[0]
        idx.append(rng.choice(ci,min(200,len(ci)),replace=False))
    keep=np.concatenate(idx)
    Xk=torch.tensor(Xtr[keep],device=dev,dtype=torch.float32); yk=torch.tensor(ytr[keep],device=dev)
    oh=torch.zeros(len(yk),ncls,device=dev); oh[torch.arange(len(yk)),yk]=1.0
    Xte_t=torch.tensor(Xte,device=dev,dtype=torch.float32); yte_t=torch.tensor(yte,device=dev)
    n_k=len(Xk); n_te=Xte_t.size(0)
    sub=Xk[rng.choice(n_k,min(4000,n_k),replace=False)]
    pd=torch.cdist(sub,sub); d=torch.quantile(pd[pd>0][:2_000_000],0.10).item(); d2=d*d
    @torch.no_grad()
    def neff_far(sig,nqq=nq):
        q=Xte_t[rng.choice(n_te,min(nqq,n_te),replace=False)]; out=[]
        for i in range(0,q.size(0),500):
            D2=torch.cdist(q[i:i+500],Xk).pow(2); fm=(D2>d2).float(); w=Ksig(sig,D2)*fm
            s1=w.sum(1); s2=(w*w).sum(1); out.append((s1*s1/(s2+1e-30)).cpu().numpy())
        return np.concatenate(out)
    @torch.no_grad()
    def tail_over_eps(sig,nqq=nq):
        q=Xte_t[rng.choice(n_te,min(nqq,n_te),replace=False)]; out=[]
        for i in range(0,q.size(0),500):
            D2=torch.cdist(q[i:i+500],Xk).pow(2); fm=(D2>d2).float(); out.append((Ksig(sig,D2)*fm).sum(1).cpu().numpy())
        return np.concatenate(out)/EPS
    @torch.no_grad()
    def acc(sig,bs=2000):
        c=0
        for i in range(0,n_te,bs):
            D2=torch.cdist(Xte_t[i:i+bs],Xk).pow(2); pr=(Ksig(sig,D2)@oh).argmax(1); c+=(pr==yte_t[i:i+bs]).sum().item()
        return c/n_te
    # sigma*
    A_mean=float(neff_far(d/np.sqrt(2*np.log(1/EPS))).mean())
    sig_star=d/np.sqrt(2*np.log(A_mean/EPS))
    # labelled grid search (30-pt log grid, argmax accuracy)
    grid=np.logspace(np.log10(0.10*d),np.log10(3.0*d),36)
    accs=[acc(s) for s in grid]; gi=int(np.argmax(accs)); sig_grid=float(grid[gi]); acc_grid=accs[gi]
    acc_star=acc(sig_star)
    # ---- Sheather-Jones (solve-the-equation plug-in), per-dim median on subsample ----
    def sj_1d(x):
        x=np.sort(x); n=len(x); s=x.std()+1e-9; IQR=np.subtract(*np.percentile(x,[75,25]))
        sig_hat=min(s,IQR/1.349) if IQR>0 else s
        # normal-reference pilot then one SJ fixed-point step (STE)
        a=0.920*sig_hat*n**(-1/7.0); b=0.912*sig_hat*n**(-1/9.0)
        xi=x[:,None]-x[None,:]
        def phi6(u): return (u**6-15*u**4+45*u**2-15)*np.exp(-u*u/2)/np.sqrt(2*np.pi)
        def phi4(u): return (u**4-6*u**2+3)*np.exp(-u*u/2)/np.sqrt(2*np.pi)
        TD=-(phi6(xi/b)).sum()/(n*(n-1)*b**7)
        SD=(phi4(xi/a)).sum()/(n*(n-1)*a**5)
        if SD<=0 or TD==0: return 1.06*sig_hat*n**(-1/5.0)
        alpha=1.357*(abs(SD/TD))**(1/7.0)*1.0
        h=(1.0/(2*np.sqrt(np.pi)*n*abs(SD)))**(1/5.0)
        return float(h if np.isfinite(h) and h>0 else 1.06*sig_hat*n**(-1/5.0))
    dsub=Xtr[rng.choice(len(Xtr),min(2000,len(Xtr)),replace=False)]
    ncheck=min(dim,64); cols=rng.choice(dim,ncheck,replace=False)
    sj=float(np.median([sj_1d(dsub[:,j]) for j in cols]))
    # ---- GP marginal-likelihood: argmax LML over the same grid, one-hot targets ----
    @torch.no_grad()
    def lml(sig,Xs,Ys):
        D2=torch.cdist(Xs,Xs).pow(2); K=Ksig(sig,D2)+1e-4*torch.eye(Xs.size(0),device=dev)
        try: L=torch.linalg.cholesky(K)
        except Exception: return -1e18
        logdet=2*torch.log(torch.diagonal(L)).sum()
        al=torch.cholesky_solve(Ys,L); quad=(Ys*al).sum()
        return float(-0.5*quad-0.5*Ys.size(1)*logdet)
    ns=min(1200,n_k); si=rng.choice(n_k,ns,replace=False)
    Xs=Xk[si]; Ys=oh[si]
    lmls=[lml(s,Xs,Ys) for s in grid]; sig_gpml=float(grid[int(np.argmax(lmls))])
    rows={}
    for tag,sig in [("sheather_jones",sj),("gp_ml",sig_gpml),("sigma_star",sig_star),("grid_labels",sig_grid)]:
        te=tail_over_eps(sig)
        rows[tag]={"sigma":float(sig),"tail_over_eps_mean":float(te.mean()),"tail_over_eps_max":float(te.max()),
                   "frac_over_1":float((te>1).mean()),"acc":acc(sig)}
    # ---- label-free A2 diagnostic: twilight-annulus weight fraction at sigma* ----
    @torch.no_grad()
    def a2_ambiguity(sig,kappa=1.25,nqq=nq):
        q=Xte_t[rng.choice(n_te,min(nqq,n_te),replace=False)]; amb=[]
        lo=(d/kappa)**2; hi=(d*kappa)**2
        for i in range(0,q.size(0),500):
            D2=torch.cdist(q[i:i+500],Xk).pow(2); w=Ksig(sig,D2)
            tw=((D2>=lo)&(D2<=hi)).float()*w
            amb.append((tw.sum(1)/(w.sum(1)+1e-30)).cpu().numpy())
        return float(np.concatenate(amb).mean())
    ambiguity=a2_ambiguity(sig_star)
    return {"name":name,"dim":int(dim),"d":float(d),"A_mean":A_mean,
            "sigma_star":float(sig_star),"sigma_grid":sig_grid,
            "acc_star":float(acc_star),"acc_grid":float(acc_grid),
            "gap_grid_minus_star":float((acc_grid-acc_star)*100),
            "a2_ambiguity":ambiguity,"baselines":rows}

OUT={"eps":EPS,"fields":[]}
# CIFAR backbones
for feat,name in [("/workspace/cifar_feats.npz","resnet18"),
                  ("/workspace/cifar_feats_r50.npz","resnet50"),
                  ("/workspace/cifar_feats_vit.npz","vit")]:
    z=np.load(feat); r=evaluate(z["Xtr"],z["Ytr"],z["Xte"],z["Yte"],name,z["Xtr"].shape[1],ncls=100)
    OUT["fields"].append(r); print(f"[{name}] sig*={r['sigma_star']:.3f} grid={r['sigma_grid']:.3f} gap={r['gap_grid_minus_star']:.2f} A2amb={r['a2_ambiguity']:.4f} SJ={r['baselines']['sheather_jones']['sigma']:.3f} GPML={r['baselines']['gp_ml']['sigma']:.3f}",flush=True)
# public sklearn datasets
from sklearn.datasets import load_iris,load_wine,load_breast_cancer,load_digits
from sklearn.model_selection import train_test_split
pub=[("iris",load_iris),("wine",load_wine),("breast_cancer",load_breast_cancer),("digits",load_digits)]
try:
    from sklearn.datasets import fetch_covtype
    pub.append(("covertype",lambda: fetch_covtype()))
except Exception as e: print("covtype skip",e)
for name,loader in pub:
    D=loader(); X=D.data.astype("float32"); y=D.target.astype("int64")
    if name=="covertype":
        y=y-1; s=rng.choice(len(X),20000,replace=False); X=X[s]; y=y[s]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    r=evaluate(Xtr,ytr,Xte,yte,name,X.shape[1])
    OUT["fields"].append(r); print(f"[{name}] sig*={r['sigma_star']:.3f} grid={r['sigma_grid']:.3f} gap={r['gap_grid_minus_star']:.2f} A2amb={r['a2_ambiguity']:.4f}",flush=True)

json.dump(OUT,open("extra_baselines_a2.json","w"),indent=2)
# correlation: does label-free ambiguity predict the labelled gap?
amb=np.array([f["a2_ambiguity"] for f in OUT["fields"]]); gap=np.array([f["gap_grid_minus_star"] for f in OUT["fields"]])
r_pear=float(np.corrcoef(amb,gap)[0,1])
json.dump({"pearson_amb_vs_gap":r_pear,"n_fields":len(OUT["fields"])},open("a2_correlation.json","w"))
print(f"A2 diagnostic: pearson(ambiguity, grid-star gap) = {r_pear:.3f} over {len(OUT['fields'])} fields")
print(f"WROTE extra_baselines_a2.json  {time.time()-t0:.1f}s")
