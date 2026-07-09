
"""Class-incremental growing-field regime: does sigma* re-certification matter as CLASSES arrive?

RESULT: NULL (this growth mode does not reproduce the density-growth effect). Classes are added
in stages (10 -> 100) at FIXED examples-per-class, so the field grows in class count but not in
local density. Measured outcome (see grow_classinc.json): the labelled-grid bandwidth fixed once
on the initial 10-class field STAYS within budget throughout (gridfix tail/eps = 0.01, 0.03, 0.09,
0.20, 0.23 across stages -- never violates) and ties sigma* on accuracy at every stage (+0.1 pts
at 100 classes). Mechanistically, adding classes at fixed density barely moves the field geometry
(d drifts only -3.5%, sigma* shrinks only 6.26->5.43), so a fixed bandwidth stays valid. The
paper's growing-field advantage is DENSITY-driven (tuned-on-sparse / deployed-on-dense mismatch),
NOT class-count-driven. This script documents that scope boundary; sigma* still holds its budget
and tracks the per-stage oracle (-0.2) here, it simply has no fixed-bandwidth failure to beat.

Protocol (standard class-incremental):
- Stage k exposes classes [0 .. Ck). Field = all train features of seen classes (fixed nper/class
  so the growth is purely in class count, isolating the class-incremental axis).
- Evaluate on the test features of seen classes only (the deployed task at stage k).
- Labelled grid search ("grid (labels)") is fixed ONCE on the initial-stage field (few classes).
- Per-stage oracle grid re-tunes sigma each stage using labels (reference upper bound, not deployable).
"""
import numpy as np, torch, json
def log(*a): print(*a, flush=True)
DEV="cuda"; rng=np.random.default_rng(0); EPS=0.05
z=np.load("/workspace/cifar_feats.npz"); Xtr,Ytr,Xte,Yte=z["Xtr"],z["Ytr"],z["Xte"],z["Yte"]
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
Xtr_t=torch.tensor(Xtr,device=DEV,dtype=torch.float32); Ytr_t=torch.tensor(Ytr,device=DEV)
Xte_t=torch.tensor(Xte,device=DEV,dtype=torch.float32); Yte_t=torch.tensor(Yte,device=DEV)

NPER=120                     # examples/class held fixed -> growth is purely in class count
CLASS_STAGES=[10,25,50,75,100]   # classes seen so far (arriving over time)
def Ksig(s,r): return torch.exp(-r/(2*s*s))

# Precompute a fixed nper-per-class training pool (same members regardless of stage)
pool_idx={c: rng.choice(np.where(Ytr==c)[0], NPER, replace=False) for c in range(100)}

def build(nclasses):
    keep=np.concatenate([pool_idx[c] for c in range(nclasses)])
    Xk_t=Xtr_t[keep]; Yk=Ytr[keep]
    Yk_t=torch.tensor(Yk,device=DEV)
    oh=torch.zeros(len(Yk),nclasses,device=DEV); oh[torch.arange(len(Yk)),Yk_t]=1.0
    # test mask: only classes seen so far
    te_mask=(Yte<nclasses)
    Xte_s=Xte_t[te_mask]; Yte_s=Yte_t[te_mask]
    return Xk_t, oh, Yk_t, Xte_s, Yte_s

def geom(Xk_t):
    m=min(4000,len(Xk_t)); s=Xk_t[rng.choice(len(Xk_t),m,replace=False)]
    D=torch.cdist(s,s); return torch.quantile(D[D>0][:2_000_000],0.10).item()

@torch.no_grad()
def measure_A(Xk_t,Xq,d,sig,nq=3000):
    q=Xq[rng.choice(Xq.size(0),min(nq,Xq.size(0)),replace=False)]; v=[]
    for i in range(0,q.size(0),500):
        D2=torch.cdist(q[i:i+500],Xk_t).pow(2); fm=D2>d*d
        w=Ksig(sig,D2)*fm; s=w.sum(1); s2=(w*w).sum(1); v.append(((s*s)/(s2+1e-30)).cpu().numpy())
    return float(np.concatenate(v).mean())

@torch.no_grad()
def true_tail(Xk_t,Xq,d,sig,nq=3000):
    q=Xq[rng.choice(Xq.size(0),min(nq,Xq.size(0)),replace=False)]; t=[]
    for i in range(0,q.size(0),500):
        D2=torch.cdist(q[i:i+500],Xk_t).pow(2); fm=D2>d*d; t.append((Ksig(sig,D2)*fm).sum(1).cpu().numpy())
    return float(np.concatenate(t).mean())

@torch.no_grad()
def acc(Xk_t,oh,Xq,Yq,sig,bs=2000):
    inv=1/(2*sig*sig); c=0
    for i in range(0,Xq.size(0),bs):
        d2=torch.cdist(Xq[i:i+bs],Xk_t).pow(2); c+=((torch.exp(-d2*inv)@oh).argmax(1)==Yq[i:i+bs]).sum().item()
    return c/Xq.size(0)

def grid_sigma(Xk_t,oh,Xq,Yq):
    grid=np.geomspace(0.5,40,30)
    return grid[int(np.argmax([acc(Xk_t,oh,Xq,Yq,s) for s in grid]))]

# Fix labelled grid search on the INITIAL few-class field
Xk0,oh0,Yk0,Xte0,Yte0=build(CLASS_STAGES[0]); d0=geom(Xk0)
sigma_grid_fixed=grid_sigma(Xk0,oh0,Xte0,Yte0)
log(f"grid sigma fixed on initial {CLASS_STAGES[0]}-class field: {sigma_grid_fixed:.3f}")
rows=[]
for nc in CLASS_STAGES:
    Xk_t,oh,Yk_t,Xq,Yq=build(nc); d=geom(Xk_t)
    A=measure_A(Xk_t,Xq,d,d/np.sqrt(2*np.log(1/EPS)))
    sstar=d/np.sqrt(2*np.log(A/EPS))
    s_oracle=grid_sigma(Xk_t,oh,Xq,Yq)
    row=dict(nclasses=nc,n=len(Xk_t),d=d,A=A,sigma_star=float(sstar),
             star_tail=true_tail(Xk_t,Xq,d,sstar)/EPS, star_acc=acc(Xk_t,oh,Xq,Yq,sstar),
             gridfix_sigma=float(sigma_grid_fixed), gridfix_tail=true_tail(Xk_t,Xq,d,sigma_grid_fixed)/EPS,
             gridfix_acc=acc(Xk_t,oh,Xq,Yq,sigma_grid_fixed),
             oracle_sigma=float(s_oracle), oracle_acc=acc(Xk_t,oh,Xq,Yq,s_oracle))
    rows.append(row)
    log(f" {nc:3d} cls n={row['n']:5d}: sigma*={sstar:.2f} tail={row['star_tail']:.2f} acc={row['star_acc']:.3f} | "
        f"gridfix={sigma_grid_fixed:.2f} tail={row['gridfix_tail']:8.2f} acc={row['gridfix_acc']:.3f} | "
        f"oracle_acc={row['oracle_acc']:.3f} | star-gridfix={row['star_acc']-row['gridfix_acc']:+.3f}")
json.dump(dict(eps=EPS,nper=NPER,sigma_grid_fixed=float(sigma_grid_fixed),stages=rows),
          open("grow_classinc.json","w"),indent=2)
log("DONE")
