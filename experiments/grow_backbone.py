
"""Density-growth regime (grow2 mechanism) parameterized by backbone feature file.
sigma* re-certifies a densifying field with zero labels where a grid-search sigma fixed on the
sparse early field drifts past the budget. Same protocol as the ResNet-18 flagship growth curve.
"""
import numpy as np, torch, json, os, sys
def log(*a): print(*a,flush=True)
DEV="cuda"; rng=np.random.default_rng(0); EPS=0.05
FEATS=os.environ.get("FEATS","/workspace/cifar_feats.npz")
TAG=os.environ.get("TAG","r18")
z=np.load(FEATS); Xtr,Ytr,Xte,Yte=z["Xtr"],z["Ytr"],z["Xte"],z["Yte"]
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
Xte_t=torch.tensor(Xte,device=DEV,dtype=torch.float32); Yte_t=torch.tensor(Yte,device=DEV)
NCLS=100
def Ksig(s,r): return torch.exp(-r/(2*s*s))
per_class_stages=[3,8,20,50,120]
def build(nper):
    keep=np.concatenate([rng.choice(np.where(Ytr==c)[0],nper,replace=False) for c in range(NCLS)])
    Xk=Xtr[keep]; Yk=Ytr[keep]
    Xk_t=torch.tensor(Xk,device=DEV,dtype=torch.float32); Yk_t=torch.tensor(Yk,device=DEV)
    oh=torch.zeros(len(Yk),NCLS,device=DEV); oh[torch.arange(len(Yk)),Yk_t]=1.0
    return Xk_t,oh
def geom(Xk_t):
    m=min(4000,len(Xk_t)); s=Xk_t[rng.choice(len(Xk_t),m,replace=False)]
    D=torch.cdist(s,s); return torch.quantile(D[D>0][:2_000_000],0.10).item()
@torch.no_grad()
def measure_A(Xk_t,d,sig,nq=3000):
    q=Xte_t[rng.choice(Xte_t.size(0),nq,replace=False)]; v=[]
    for i in range(0,nq,500):
        D2=torch.cdist(q[i:i+500],Xk_t).pow(2); fm=D2>d*d
        w=Ksig(sig,D2)*fm; s=w.sum(1); s2=(w*w).sum(1); v.append(((s*s)/(s2+1e-30)).cpu().numpy())
    return float(np.concatenate(v).mean())
@torch.no_grad()
def true_tail(Xk_t,d,sig,nq=3000):
    q=Xte_t[rng.choice(Xte_t.size(0),nq,replace=False)]; t=[]
    for i in range(0,nq,500):
        D2=torch.cdist(q[i:i+500],Xk_t).pow(2); fm=D2>d*d; t.append((Ksig(sig,D2)*fm).sum(1).cpu().numpy())
    return float(np.concatenate(t).mean())
@torch.no_grad()
def acc(Xk_t,oh,sig,bs=2000):
    inv=1/(2*sig*sig); c=0
    for i in range(0,Xte_t.size(0),bs):
        d2=torch.cdist(Xte_t[i:i+bs],Xk_t).pow(2); c+=((torch.exp(-d2*inv)@oh).argmax(1)==Yte_t[i:i+bs]).sum().item()
    return c/Xte_t.size(0)
def grid_sigma(Xk_t,oh):
    grid=np.geomspace(0.5,40,30)
    return grid[int(np.argmax([acc(Xk_t,oh,s) for s in grid]))]
Xk0,oh0=build(per_class_stages[0]); d0=geom(Xk0)
sigma_grid_fixed=grid_sigma(Xk0,oh0)
log(f"[{TAG}] grid sigma fixed on sparse {per_class_stages[0]}/class field: {sigma_grid_fixed:.3f}")
rows=[]
for nper in per_class_stages:
    Xk_t,oh=build(nper); d=geom(Xk_t)
    A=measure_A(Xk_t,d,d/np.sqrt(2*np.log(1/EPS)))
    sstar=d/np.sqrt(2*np.log(A/EPS))
    s_oracle=grid_sigma(Xk_t,oh)
    row=dict(nper=nper,n=len(Xk_t),d=d,A=A,sigma_star=float(sstar),
             star_tail=true_tail(Xk_t,d,sstar)/EPS,star_acc=acc(Xk_t,oh,sstar),
             gridfix_sigma=float(sigma_grid_fixed),gridfix_tail=true_tail(Xk_t,d,sigma_grid_fixed)/EPS,gridfix_acc=acc(Xk_t,oh,sigma_grid_fixed),
             oracle_sigma=float(s_oracle),oracle_acc=acc(Xk_t,oh,s_oracle))
    rows.append(row)
    log(f" [{TAG}] {nper:3d}/cls n={row['n']:5d}: s*={sstar:.2f} tail={row['star_tail']:.2f} acc={row['star_acc']:.3f} | "
        f"gridfix={sigma_grid_fixed:.2f} tail={row['gridfix_tail']:7.2f} acc={row['gridfix_acc']:.3f} | orac={row['oracle_acc']:.3f} "
        f"| s*-gridfix={row['star_acc']-row['gridfix_acc']:+.3f}")
json.dump(dict(tag=TAG,eps=EPS,sigma_grid_fixed=float(sigma_grid_fixed),stages=rows),open(f"grow_{TAG}.json","w"),indent=2)
log("DONE")
