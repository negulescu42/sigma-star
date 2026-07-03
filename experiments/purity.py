
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
A=float(neff_far(sig_pair).mean()); ss=d/np.sqrt(2*np.log(A/EPS))
# weighted near-set purity at sigma* : fraction of NEAR kernel mass on same-class sources
chance=1.0/100
@torch.no_grad()
def wpurity(nq=3000):
    idx=rng.choice(n_te,nq,replace=False); q=Xte_t[idx]; qy=yte_t[idx]; wp=[]; up=[]
    for i in range(0,nq,500):
        D2=torch.cdist(q[i:i+500],Xk).pow(2); near=(D2<=d2)
        w=Ksig(ss,D2)*near.float()
        same=(yk.unsqueeze(0)==qy[i:i+500].unsqueeze(1)).float()
        wm=(w*same).sum(1)/w.sum(1).clamp(min=1e-9); wp.append(wm.cpu().numpy())
        cnt=near.sum(1).clamp(min=1); up.append(((near.float()*same).sum(1)/cnt).cpu().numpy())
    return np.concatenate(wp), np.concatenate(up)
wp,up=wpurity()
# selective prediction baseline: softmax-margin threshold on the SAME kernel vote at sigma*,
# sweep coverage; compare accuracy-on-covered vs the certificate's covered set.
@torch.no_grad()
def sel_curve(nq=10000):
    idx=rng.choice(n_te,nq,replace=False); q=Xte_t[idx]; qy=yte_t[idx]
    marg=[]; corr=[]
    for i in range(0,nq,1000):
        D2=torch.cdist(q[i:i+1000],Xk).pow(2); sc=Ksig(ss,D2)@oh
        top2=torch.topk(sc,2,1).values; m=(top2[:,0]-top2[:,1])
        pred=sc.argmax(1); marg.append(m.cpu().numpy()); corr.append((pred==qy[i:i+1000]).cpu().numpy())
    marg=np.concatenate(marg); corr=np.concatenate(corr)
    # accuracy at coverage levels by margin threshold
    order=np.argsort(-marg); out={}
    for cov in [0.27,0.5,0.75,1.0]:
        k=int(cov*len(marg)); out[f"cov{cov}"]=float(corr[order[:k]].mean())
    return out
sel=sel_curve()
res={"name":NAME,"d10":d,"sigma_star":float(ss),"chance_purity":chance,
     "weighted_near_purity":{"mean":float(wp.mean()),"median":float(np.median(wp))},
     "unweighted_near_purity":{"mean":float(up.mean()),"median":float(np.median(up))},
     "selective_margin_acc":sel}
print(f"[{NAME}] wpur mean={wp.mean():.3f} med={np.median(wp):.3f} | upur mean={up.mean():.3f} | chance={chance:.3f} | enrich_w={wp.mean()/chance:.1f}x")
print(f"[{NAME}] selective acc @cov: {sel}")
json.dump(res,open(f"purity_{NAME}.json","w"),indent=2)
print(f"WROTE purity_{NAME}.json {time.time()-t0:.1f}s")
