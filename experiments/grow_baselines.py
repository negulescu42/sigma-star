
"""WP5.2: recompute label-free baselines (median, kNN) at EACH densification stage,
alongside sigma* and the stale labelled-grid sigma. All label-free methods receive the
UPDATED geometry at every stage; only sigma* is designed to hold the interference budget.
Runs the 3-backbone density sweep (R18/R50/ViT features)."""
import numpy as np, torch, json
def log(*a): print(*a,flush=True)
DEV="cuda"; rng=np.random.default_rng(0); EPS=0.05
FEATS={"r18":"/workspace/cifar_feats.npz","r50":"/workspace/cifar_feats_r50.npz","vit":"/workspace/cifar_feats_vit.npz"}
stages=[10,25,50,75,100]

def Ksig(s,r): return torch.exp(-r/(2*s*s))

def run_backbone(tag,path):
    z=np.load(path); Xtr,Ytr,Xte,Yte=z["Xtr"],z["Ytr"],z["Xte"],z["Yte"]
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    Xte_t=torch.tensor(Xte,device=DEV,dtype=torch.float32); Yte_t=torch.tensor(Yte,device=DEV)
    def build_field(nc):
        cls=np.arange(nc)
        keep=np.concatenate([rng.choice(np.where(Ytr==c)[0],200,replace=False) for c in cls])
        Xk=Xtr[keep]; Yk=Ytr[keep]
        Xk_t=torch.tensor(Xk,device=DEV,dtype=torch.float32); Yk_t=torch.tensor(Yk,device=DEV)
        oh=torch.zeros(len(Yk),100,device=DEV); oh[torch.arange(len(Yk)),Yk_t]=1.0
        temask=torch.tensor(np.isin(Yte,cls),device=DEV)
        return Xk_t,Yk_t,oh,temask
    def geom(Xk_t):
        m=min(4000,len(Xk_t)); s=Xk_t[rng.choice(len(Xk_t),m,replace=False)]
        D=torch.cdist(s,s); d=torch.quantile(D[D>0][:2_000_000],0.10).item(); return d,s
    @torch.no_grad()
    def Aref(Xk_t,d,sref,nq=2000):
        q=Xte_t[rng.choice(Xte_t.size(0),nq,replace=False)]; v=[]
        for i in range(0,nq,500):
            D2=torch.cdist(q[i:i+500],Xk_t).pow(2); fm=D2>d*d
            w=Ksig(sref,D2)*fm; s=w.sum(1); s2=(w*w).sum(1); v.append(((s*s)/(s2+1e-30)).cpu().numpy())
        return float(np.concatenate(v).mean())
    @torch.no_grad()
    def tail_over_eps(Xk_t,d,sig,nq=2000):
        q=Xte_t[rng.choice(Xte_t.size(0),nq,replace=False)]; t=[]
        for i in range(0,nq,500):
            D2=torch.cdist(q[i:i+500],Xk_t).pow(2); fm=D2>d*d; t.append((Ksig(sig,D2)*fm).sum(1).cpu().numpy())
        return float(np.concatenate(t).mean())/EPS
    @torch.no_grad()
    def acc(Xk_t,oh,sig,temask,bs=2000):
        inv=1/(2*sig*sig); idx=torch.where(temask)[0]; Xq=Xte_t[idx]; Yq=Yte_t[idx]; c=0
        for i in range(0,Xq.size(0),bs):
            d2=torch.cdist(Xq[i:i+bs],Xk_t).pow(2); c+=((torch.exp(-d2*inv)@oh).argmax(1)==Yq[i:i+bs]).sum().item()
        return c/Xq.size(0)
    @torch.no_grad()
    def median_scale(Xk_t,nsub=3000):
        m=min(nsub,len(Xk_t)); s=Xk_t[rng.choice(len(Xk_t),m,replace=False)]
        D=torch.cdist(s,s); return torch.median(D[D>0]).item()
    @torch.no_grad()
    def knn_scale(Xk_t,k=7,nsub=3000):
        m=min(nsub,len(Xk_t)); s=Xk_t[rng.choice(len(Xk_t),m,replace=False)]
        D=torch.cdist(s,Xk_t); kd=torch.topk(D,k+1,largest=False).values[:,-1]  # k-th NN dist
        return kd.mean().item()
    # stale labelled-grid sigma, fixed once on stage 0
    Xk0,Yk0,oh0,tem0=build_field(stages[0]); d0,_=geom(Xk0)
    grid=np.geomspace(0.5,max(30,d0*3),30)
    sigma_grid_fixed=grid[int(np.argmax([acc(Xk0,oh0,float(s),tem0) for s in grid]))]
    log(f"[{tag}] labelled-grid sigma fixed on {stages[0]}-class: {sigma_grid_fixed:.3f}")
    rows=[]
    for nc in stages:
        Xk_t,Yk_t,oh,tem=build_field(nc); d,_=geom(Xk_t)
        Apair=Aref(Xk_t,d,d/np.sqrt(2*np.log(1/EPS)))
        sstar=d/np.sqrt(2*np.log(Apair/EPS))
        s_med=median_scale(Xk_t); s_knn=knn_scale(Xk_t)   # label-free, RECOMPUTED at THIS stage
        row=dict(nc=nc,n=len(Xk_t),d=d,A=Apair,
                 sstar=float(sstar), star_tail=tail_over_eps(Xk_t,d,sstar), star_acc=acc(Xk_t,oh,sstar,tem),
                 gridfix_sigma=float(sigma_grid_fixed), gridfix_tail=tail_over_eps(Xk_t,d,sigma_grid_fixed), gridfix_acc=acc(Xk_t,oh,sigma_grid_fixed,tem),
                 median_sigma=float(s_med), median_tail=tail_over_eps(Xk_t,d,s_med), median_acc=acc(Xk_t,oh,s_med,tem),
                 knn_sigma=float(s_knn), knn_tail=tail_over_eps(Xk_t,d,s_knn), knn_acc=acc(Xk_t,oh,s_knn,tem))
        rows.append(row)
        log(f"[{tag}] {nc:3d}cls n={row['n']:5d} | s*={sstar:5.2f} t/e={row['star_tail']:7.2f} a={row['star_acc']:.3f}"
            f" | grid t/e={row['gridfix_tail']:8.2f} a={row['gridfix_acc']:.3f}"
            f" | med s={s_med:5.2f} t/e={row['median_tail']:8.1f} a={row['median_acc']:.3f}"
            f" | knn s={s_knn:5.2f} t/e={row['knn_tail']:8.1f} a={row['knn_acc']:.3f}")
    return dict(tag=tag,eps=EPS,sigma_grid_fixed=float(sigma_grid_fixed),stages=rows)

out={}
import os
for tag,path in FEATS.items():
    if os.path.exists(path):
        out[tag]=run_backbone(tag,path)
    else:
        log(f"[{tag}] MISSING {path}, skipping")
json.dump(out,open("grow_baselines.json","w"),indent=1)
log("DONE", list(out.keys()))
