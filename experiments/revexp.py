
import numpy as np, json, time, sys, torch
t0=time.time(); dev="cuda" if torch.cuda.is_available() else "cpu"
FEAT=sys.argv[1]; NAME=sys.argv[2]; EPS=0.05; rng=np.random.default_rng(0)
z=np.load(FEAT); Xtr,ytr,Xte,yte=z["Xtr"],z["Ytr"],z["Xte"],z["Yte"]
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
keep=np.concatenate([rng.choice(np.where(ytr==c)[0],200,replace=False) for c in range(100)])
Xk=torch.tensor(Xtr[keep],device=dev,dtype=torch.float32); yk=torch.tensor(ytr[keep],device=dev)
oh=torch.zeros(len(yk),100,device=dev); oh[torch.arange(len(yk)),yk]=1.0
Xte_t=torch.tensor(Xte,device=dev,dtype=torch.float32); yte_t=torch.tensor(yte,device=dev)
n_k=len(Xk); n_te=Xte_t.size(0)
sub=Xk[rng.choice(n_k,4000,replace=False)]
pdist=torch.cdist(sub,sub); pv=pdist[pdist>0]
def dpct(p): return float(torch.quantile(pv[:2_000_000],p/100.0))
d=dpct(10); d2=d*d
Ksig=lambda s,r: torch.exp(-r/(2*s*s))
@torch.no_grad()
def neff_far(sig,nq=4000):
    q=Xte_t[rng.choice(n_te,nq,replace=False)]; out=[]
    for i in range(0,nq,500):
        D2=torch.cdist(q[i:i+500],Xk).pow(2); fm=(D2>d2).float(); w=Ksig(sig,D2)*fm
        s1=w.sum(1); s2=(w*w).sum(1); out.append((s1*s1/(s2+1e-30)).cpu().numpy())
    return np.concatenate(out)
@torch.no_grad()
def tail_over_eps(sig,dd2=None,nq=4000):
    dd2=d2 if dd2 is None else dd2
    q=Xte_t[rng.choice(n_te,nq,replace=False)]; out=[]
    for i in range(0,nq,500):
        D2=torch.cdist(q[i:i+500],Xk).pow(2); fm=(D2>dd2).float(); out.append((Ksig(sig,D2)*fm).sum(1).cpu().numpy())
    return np.concatenate(out)/EPS
@torch.no_grad()
def acc(sig,bs=2000):
    c=0
    for i in range(0,n_te,bs):
        D2=torch.cdist(Xte_t[i:i+bs],Xk).pow(2); c+=((Ksig(sig,D2)@oh).argmax(1)==yte_t[i:i+bs]).sum().item()
    return c/n_te
def sig_star(A,dd,eps=EPS): return dd/np.sqrt(2*np.log(A/eps))
# ---- our rule
A_mean=float(neff_far(sig_star( float(neff_far(d/np.sqrt(2*np.log(1/EPS)))[0:1].mean()) if False else 12000, d)).mean()) if False else None
# proper: measure A at sigma_pair
sig_pair=d/np.sqrt(2*np.log(1/EPS)); A=float(neff_far(sig_pair).mean()); ss=sig_star(A,d)
# ---- median heuristic: sigma s.t. RBF uses median pairwise distance => sigma_med = median/ (sqrt? ) use sigma=median dist
sig_med=float(torch.median(pv))
# ---- kNN local scaling: mean distance to k-th NN among sources (k=7)
@torch.no_grad()
def knn_sigma(k=7):
    tot=0.0; nb=0
    for i in range(0,n_k,1000):
        D=torch.cdist(Xk[i:i+1000],Xk); D,_=torch.sort(D,1); tot+=D[:,k].sum().item(); nb+=D.size(0)
    return tot/nb
sig_knn=knn_sigma(7)
res={"name":NAME,"eps":EPS,"d10":d}
for tag,sig in [("star",ss),("median",sig_med),("knn7",sig_knn)]:
    te=tail_over_eps(sig)
    res[tag]={"sigma":float(sig),"tail_mean":float(te.mean()),"tail_max":float(te.max()),
              "frac_over_1":float((te>1).mean()),"acc":acc(sig)}
    print(f"[{NAME} {tag}] sig={sig:.3f} tail_mean={te.mean():.3f} tail_max={te.max():.3f} over1={float((te>1).mean()):.4f} acc={res[tag]['acc']*100:.2f}")
# ---- sensitivity: eps x d-percentile grid (accuracy + tail)
sens=[]
for pe in [1,5,10,20]:
    dd=dpct(pe); dd2=dd*dd
    for ep in [0.01,0.05,0.1,0.2]:
        sp=dd/np.sqrt(2*np.log(1/ep)); Ai=float(neff_far(sp).mean()); si=dd/np.sqrt(2*np.log(Ai/ep))
        sens.append({"dpct":pe,"eps":ep,"sigma":float(si),"acc":acc(si)})
res["sensitivity"]=sens
sa=[s["acc"] for s in sens]; print(f"[{NAME} sens] acc range {min(sa)*100:.2f}-{max(sa)*100:.2f} span {(max(sa)-min(sa))*100:.2f}pt")
# ---- A2 purity: fraction of within-d neighbours sharing query class (query=test point, neighbours=sources)
@torch.no_grad()
def purity(nq=2000):
    q=Xte_t[rng.choice(n_te,nq,replace=False)]; qy=yte_t[rng.choice(n_te,nq,replace=False)]
    # use aligned sample
    idx=rng.choice(n_te,nq,replace=False); q=Xte_t[idx]; qy=yte_t[idx]; fr=[]
    for i in range(0,nq,500):
        D=torch.cdist(q[i:i+500],Xk); near=D<d
        same=(yk.unsqueeze(0)==qy[i:i+500].unsqueeze(1))
        cnt=near.sum(1).clamp(min=1); fr.append(((near&same).sum(1)/cnt).cpu().numpy())
    return np.concatenate(fr)
pur=purity(); res["A2_purity"]={"mean":float(pur.mean()),"median":float(np.median(pur)),"frac_gt_half":float((pur>0.5).mean())}
print(f"[{NAME} A2] within-d class purity mean={pur.mean():.3f} median={np.median(pur):.3f} frac>0.5={float((pur>0.5).mean()):.3f}")
json.dump(res,open(f"revexp_{NAME}.json","w"),indent=2)
print(f"WROTE revexp_{NAME}.json {time.time()-t0:.1f}s")
