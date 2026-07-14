
"""Packing kill-test (rigorous). Doubling-metric packing: a delta-separated set
inside a ball of radius R has at most (2R/delta)^k points (k = doubling dim).
Shell [r,r+delta] count <= (2 r_out/delta)^k - (2 r_in/delta)^k. Gaussian shell
envelope: tail(sigma) <= sum_j N_j Vmax exp(-r_in_j^2/2sigma^2). sigma*_pack =
largest sigma with tail<=eps. Report ratio to measured sigma*, and sensitivity
to k (measured k, and k+-1)."""
import numpy as np, torch, json
def log(*a): print(*a,flush=True)
DEV="cuda"; rng=np.random.default_rng(0); EPS=0.05; Vmax=1.0
FEATS={"r18":"/workspace/cifar_feats.npz","r50":"/workspace/cifar_feats_r50.npz","vit":"/workspace/cifar_feats_vit.npz"}
sstar_meas={"r18":6.0807,"r50":11.5717,"vit":7.7279}

def geom(path):
    z=np.load(path); Xtr=z["Xtr"]; Ytr=z["Ytr"]
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd
    cls=np.arange(10)
    keep=np.concatenate([rng.choice(np.where(Ytr==c)[0],200,replace=False) for c in cls])
    Xk=torch.tensor(Xtr[keep],device=DEV,dtype=torch.float32); n=len(Xk)
    D=torch.cdist(Xk,Xk); off=D[~torch.eye(n,dtype=torch.bool,device=DEV)]
    d=torch.quantile(off,0.10).item(); delta=off[off>0].min().item()
    ks=[]; idx=rng.choice(n,min(300,n),replace=False)
    for i in idx:
        di=D[i][D[i]>0].cpu().numpy()
        for R in np.quantile(di,[0.3,0.5,0.7]):
            nR=int((di<=R).sum()); nR2=int((di<=R/2).sum())
            if nR2>=1 and nR>nR2: ks.append(np.log2(nR/nR2))
    return n,d,delta,float(np.mean(ks))

def sstar_pack(d,delta,k,Jcap=500000):
    def tail(sig):
        s2=2*sig*sig; tot=0.0; prev=(2*d/delta)**k
        for j in range(Jcap):
            r_out=d+(j+1)*delta; cur=(2*r_out/delta)**k
            cnt=cur-prev; prev=cur; r_in=d+j*delta
            term=cnt*Vmax*np.exp(-(r_in*r_in)/s2); tot+=term
            if term<1e-20 and j>10: break
        return tot
    if tail(1e-3)>EPS: return 0.0
    lo,hi=1e-3,d
    while tail(hi)<EPS and hi<d*10: hi*=1.5
    for _ in range(90):
        mid=0.5*(lo+hi)
        if tail(mid)<=EPS: lo=mid
        else: hi=mid
    return lo

out={}
for tag,path in FEATS.items():
    n,d,delta,k=geom(path)
    sp=sstar_pack(d,delta,k); sm=sstar_meas[tag]
    sp_km=sstar_pack(d,delta,k+1); sp_kp=sstar_pack(d,delta,max(1.0,k-1))
    r=dict(tag=tag,n=n,d=round(d,4),delta=round(delta,6),d_over_delta=round(d/delta,1),
           kdoub=round(k,3),sstar_pack=round(sp,4),sstar_meas=sm,
           ratio=round(sp/sm,4),sstar_pack_kplus1=round(sp_km,4),sstar_pack_kminus1=round(sp_kp,4))
    out[tag]=r; log(tag,json.dumps(r))
json.dump(out,open("packing_killtest.json","w"),indent=2)
log("=== DONE ===")
