
import numpy as np, json, time, sys, torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
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
@torch.no_grad()
def neff_far(sig,nq=10000):
    q=Xte_t[rng.choice(n_te,nq,replace=False)]; out=[]
    for i in range(0,nq,500):
        D2=torch.cdist(q[i:i+500],Xk).pow(2); fm=(D2>d2).float(); w=Ksig(sig,D2)*fm
        s1=w.sum(1); s2=(w*w).sum(1); out.append((s1*s1/(s2+1e-30)).cpu().numpy())
    return np.concatenate(out)
@torch.no_grad()
def tail_dist(sig,nq=10000):
    q=Xte_t[rng.choice(n_te,nq,replace=False)]; out=[]
    for i in range(0,nq,500):
        D2=torch.cdist(q[i:i+500],Xk).pow(2); fm=(D2>d2).float(); out.append((Ksig(sig,D2)*fm).sum(1).cpu().numpy())
    return np.concatenate(out)/EPS
@torch.no_grad()
def acc(sig,bs=2000):
    c=0
    for i in range(0,n_te,bs):
        D2=torch.cdist(Xte_t[i:i+bs],Xk).pow(2); pr=(Ksig(sig,D2)@oh).argmax(1); c+=(pr==yte_t[i:i+bs]).sum().item()
    return c/n_te
@torch.no_grad()
def giant(sig,ncent=5000):
    C=Xk[rng.choice(n_k,ncent,replace=False)]; D=torch.cdist(C,C)
    Aadj=((D<3*sig)&(D>0)).cpu().numpy(); _,lab=connected_components(csr_matrix(Aadj),directed=False)
    _,cnt=np.unique(lab,return_counts=True); return float(cnt.max()/ncent)
# measure Neff distribution at sig_pair
nf=neff_far(sig_pair); A_mean=float(nf.mean()); A_p99=float(np.percentile(nf,99))
sig_star=d/np.sqrt(2*np.log(A_mean/EPS)); sig_p99=d/np.sqrt(2*np.log(A_p99/EPS))
res={"name":NAME,"dim":dim,"eps":EPS,"d":float(d),"A_mean":A_mean,"A_p99":A_p99,
     "sigma_star":float(sig_star),"sigma_p99":float(sig_p99)}
for tag,sig in [("star",sig_star),("p99",sig_p99)]:
    te=tail_dist(sig)
    res[tag]={"sigma":float(sig),"tail_mean":float(te.mean()),"tail_p95":float(np.percentile(te,95)),
              "tail_p99":float(np.percentile(te,99)),"tail_max":float(te.max()),
              "frac_over_1":float((te>1).mean()),"giant":giant(sig),"acc":acc(sig)}
    print(f"[{NAME} {tag}] sig={sig:.4f} tail max={te.max():.3f} mean={te.mean():.3f} over1={float((te>1).mean()):.4f} giant={res[tag]['giant']*100:.1f}% acc={res[tag]['acc']:.4f}")
json.dump(res,open(f"p99cert_{NAME}.json","w"),indent=2)
print(f"WROTE p99cert_{NAME}.json {time.time()-t0:.1f}s")
