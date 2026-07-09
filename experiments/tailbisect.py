#!/usr/bin/env python3
"""Label-free tail-bisection baseline vs the closed-form sigma* (reviewer request).

The certificate sigma* = d/sqrt(2 ln(A/eps)) is a ONE-SHOT closed form with an
a-priori conservativeness proof (Prop 2). A reviewer notes an obvious label-free
alternative: since Tail_sigma(y,d) is strictly increasing in sigma and needs the SAME
kernel evaluations as measuring A, one can directly BISECT sigma until the measured
mean tail hits eps -- no labels, no closed form. This script runs that baseline on the
three frozen backbones and reports, per backbone:
  sigma_star (closed form, conservative), its measured mean tail/eps and accuracy;
  sigma_bisect (largest sigma with measured mean tail/eps <= 1), its accuracy;
  grid oracle (labelled argmax over a sigma grid), its accuracy.
So we can state honestly: bisection deploys at a larger sigma (sigma* is conservative:
its tail is well below eps) and closes part of the accuracy gap, but it is an empirical
search with NO a-priori guarantee attached, whereas sigma* is one-shot and proved.
We also report bisection's worst-case (p99/max) tail so its lack of a conservativeness
proof is visible: it holds the MEAN tail at eps by construction but not the tail
distribution.
"""
import numpy as np, json, time, sys, torch
t0=time.time(); dev="cuda" if torch.cuda.is_available() else "cpu"
FEAT=sys.argv[1]; NAME=sys.argv[2]; EPS=0.05; rng=np.random.default_rng(0)
z=np.load(FEAT); Xtr,ytr,Xte,yte=z["Xtr"],z["Ytr"],z["Xte"],z["Yte"]
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
dim=Xtr.shape[1]
keep=np.concatenate([rng.choice(np.where(ytr==c)[0],200,replace=False) for c in range(100)])
Xk=torch.tensor(Xtr[keep],device=dev,dtype=torch.float32); yk=torch.tensor(ytr[keep],device=dev)
oh=torch.zeros(len(yk),100,device=dev); oh[torch.arange(len(yk)),yk]=1.0
Xte_t=torch.tensor(Xte,device=dev,dtype=torch.float32); yte_t=torch.tensor(yte,device=dev)
n_k=len(Xk); n_te=Xte_t.size(0)
sub=Xk[rng.choice(n_k,4000,replace=False)]
pd=torch.cdist(sub,sub); d=torch.quantile(pd[pd>0][:2_000_000],0.10).item(); d2=d*d
Ksig=lambda s,r: torch.exp(-r/(2*s*s))
sig_pair=d/np.sqrt(2*np.log(1/EPS))
# fixed held-out query sample reused across all sigma evals (label-free tail; labelled acc)
QIDX=rng.choice(n_te,10000,replace=False); Q=Xte_t[QIDX]; yQ=yte_t[QIDX]
@torch.no_grad()
def neff_far(sig):
    out=[]
    for i in range(0,Q.size(0),500):
        D2=torch.cdist(Q[i:i+500],Xk).pow(2); fm=(D2>d2).float(); w=Ksig(sig,D2)*fm
        s1=w.sum(1); s2=(w*w).sum(1); out.append((s1*s1/(s2+1e-30)).cpu().numpy())
    return np.concatenate(out)
@torch.no_grad()
def tail_stats(sig):
    out=[]
    for i in range(0,Q.size(0),500):
        D2=torch.cdist(Q[i:i+500],Xk).pow(2); fm=(D2>d2).float()
        out.append((Ksig(sig,D2)*fm).sum(1).cpu().numpy())
    te=np.concatenate(out)/EPS
    return float(te.mean()), float(np.percentile(te,99)), float(te.max())
@torch.no_grad()
def acc(sig,bs=2000):
    c=0
    for i in range(0,n_te,bs):
        D2=torch.cdist(Xte_t[i:i+bs],Xk).pow(2); pr=(Ksig(sig,D2)@oh).argmax(1); c+=(pr==yte_t[i:i+bs]).sum().item()
    return c/n_te
# closed-form sigma*
A_mean=float(neff_far(sig_pair).mean())
sig_star=d/np.sqrt(2*np.log(A_mean/EPS))
star_tail_mean,star_tail_p99,star_tail_max=tail_stats(sig_star)
# tail-bisection: largest sigma with measured MEAN tail/eps <= 1 (label-free)
lo,hi=sig_star, sig_star*6.0   # sigma* is conservative so bisect UPward; cap at 6x
# ensure hi overshoots
for _ in range(40):
    if tail_stats(hi)[0]>1.0: break
    hi*=1.5
for _ in range(40):
    mid=0.5*(lo+hi)
    if tail_stats(mid)[0]<=1.0: lo=mid
    else: hi=mid
sig_bisect=lo
bis_tail_mean,bis_tail_p99,bis_tail_max=tail_stats(sig_bisect)
# labelled grid oracle: argmax accuracy over sigma grid anchored to sigma*
grid=sig_star*np.array([0.5,0.7,0.85,1.0,1.2,1.5,2.0,2.5,3.0,4.0])
grid_accs=[(float(s),acc(float(s))) for s in grid]
grid_sig,grid_acc=max(grid_accs,key=lambda t:t[1])
res={"name":NAME,"dim":dim,"eps":EPS,"d":float(d),"A_mean":A_mean,
 "sigma_star":float(sig_star),"star_tail_mean":star_tail_mean,"star_tail_p99":star_tail_p99,
 "star_tail_max":star_tail_max,"star_acc":acc(sig_star),
 "sigma_bisect":float(sig_bisect),"bisect_over_star":float(sig_bisect/sig_star),
 "bis_tail_mean":bis_tail_mean,"bis_tail_p99":bis_tail_p99,"bis_tail_max":bis_tail_max,
 "bisect_acc":acc(sig_bisect),
 "grid_sigma":float(grid_sig),"grid_acc":grid_acc,"grid_accs":grid_accs}
print(f"[{NAME}] sig*={sig_star:.3f} acc={res['star_acc']:.4f} tailmean={star_tail_mean:.3f} "
      f"| bisect sig={sig_bisect:.3f}({res['bisect_over_star']:.2f}x*) acc={res['bisect_acc']:.4f} "
      f"tailmean={bis_tail_mean:.3f} p99={bis_tail_p99:.2f} max={bis_tail_max:.2f} "
      f"| grid sig={grid_sig:.3f} acc={grid_acc:.4f}")
json.dump(res,open(f"tailbisect_{NAME}.json","w"),indent=2)
print(f"WROTE tailbisect_{NAME}.json {time.time()-t0:.1f}s")
