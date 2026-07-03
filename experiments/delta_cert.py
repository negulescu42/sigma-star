
import numpy as np, json, time, sys, torch
t0=time.time(); dev="cuda" if torch.cuda.is_available() else "cpu"
FEAT=sys.argv[1]; NAME=sys.argv[2]; EPS=0.05; rng=np.random.default_rng(0)
z=np.load(FEAT); Xtr,ytr,Xte,yte=z["Xtr"],z["Ytr"],z["Xte"],z["Yte"]
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
keep=np.concatenate([rng.choice(np.where(ytr==c)[0],200,replace=False) for c in range(100)])
Xk=torch.tensor(Xtr[keep],device=dev,dtype=torch.float32)
Xte_t=torch.tensor(Xte,device=dev,dtype=torch.float32)
n_k=len(Xk); n_te=Xte_t.size(0)
sub=Xk[rng.choice(n_k,4000,replace=False)]; pv=torch.cdist(sub,sub); pv=pv[pv>0]
d=float(torch.quantile(pv[:2_000_000],0.10)); d2=d*d
Ksig=lambda s,r: torch.exp(-r/(2*s*s))
sig_pair=d/np.sqrt(2*np.log(1/EPS))
@torch.no_grad()
def neff_far(sig,nq=4000):
    q=Xte_t[rng.choice(n_te,nq,replace=False)]; out=[]
    for i in range(0,nq,500):
        D2=torch.cdist(q[i:i+500],Xk).pow(2); fm=(D2>d2).float(); w=Ksig(sig,D2)*fm
        s1=w.sum(1); s2=(w*w).sum(1); out.append((s1*s1/(s2+1e-30)).cpu().numpy())
    return np.concatenate(out)
A=float(neff_far(sig_pair).mean()); ss=d/np.sqrt(2*np.log(A/EPS)); ss2=ss*ss
# far-set size |F| and Tail(c) at representative query centers
@torch.no_grad()
def field_stats(nq=3000):
    q=Xte_t[rng.choice(n_te,nq,replace=False)]; Fsz=[]; tails=[]
    for i in range(0,nq,500):
        D2=torch.cdist(q[i:i+500],Xk).pow(2); fm=(D2>d2)
        Fsz.append(fm.sum(1).cpu().numpy())
        tails.append((Ksig(ss,D2)*fm.float()).sum(1).cpu().numpy())  # far MASS (Vmax=1)
    return np.concatenate(Fsz), np.concatenate(tails)
Fsz,tails=field_stats()
Fmax=int(Fsz.max()); Fmean=float(Fsz.mean())
Kd=float(np.exp(-d2/(2*ss2)))                       # K_sigma*(d^2) = eps/A
L=1.0*Fmax*(d/ss2)*Kd                               # Lipschitz constant, Vmax=1, worst-case |F|
# certified delta: keep sup Tail <= EPS (mass budget). Use a high-tail center (p99) to be safe.
Tc=float(np.percentile(tails,99))                   # conservative center tail (mass)
delta=(EPS-Tc)/L if L>0 else float('nan')
# typical inter-query distance for scale reference
qq=Xte_t[rng.choice(n_te,2000,replace=False)]; iqd=torch.cdist(qq,qq); iqd=iqd[iqd>0]
iq_med=float(torch.median(iqd))
res={"name":NAME,"eps":EPS,"d":d,"sigma_star":ss,"A":A,"F_mean":Fmean,"F_max":Fmax,
     "K_sigmastar_d2":Kd,"L":L,"Tail_center_p99_mass":Tc,"certified_delta":delta,
     "interquery_median_dist":iq_med,"delta_over_interquery":delta/iq_med}
print(f"[{NAME}] d={d:.2f} sig*={ss:.3f} |F|max={Fmax} K(d2)={Kd:.2e} L={L:.4f} Tc(p99 mass)={Tc:.4f} "
      f"delta={delta:.3f} interq_med={iq_med:.1f} delta/interq={delta/iq_med:.4f}")
json.dump(res,open(f"delta_{NAME}.json","w"),indent=2)
print(f"WROTE delta_{NAME}.json {time.time()-t0:.1f}s")
