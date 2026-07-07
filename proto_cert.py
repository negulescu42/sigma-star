#!/usr/bin/env python3
"""ProtoNet certificate: a prototypical network's distance-softmax readout is a Gaussian
kernel field over class prototypes with BOUNDED weights (v_k=1), so BOTH the interference-tail
certificate (Prop 1) AND the full far-set locality readout (Cor 2) transfer -- unlike attention.
Trains a real ProtoNet on Omniglot, then certifies the prototype fields on held-out episodes."""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
torch.manual_seed(20260707); np.random.seed(20260707)
EPS=0.05; DEV="cuda" if torch.cuda.is_available() else "cpu"
DATA="/workspace/cs_jobs/omniglot"

def conv_block(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2))
class Encoder(nn.Module):
    def __init__(s):
        super().__init__(); s.e=nn.Sequential(conv_block(1,64),conv_block(64,64),conv_block(64,64),conv_block(64,64))
    def forward(s,x): return s.e(x).flatten(1)   # 64-d embedding on 28x28

tf=transforms.Compose([transforms.Grayscale(),transforms.Resize((28,28)),transforms.ToTensor()])
ds=datasets.Omniglot(root=DATA,background=True,download=True,transform=tf)
# group indices by character label
by=dict()
for idx,(_,y) in enumerate(ds._flat_character_images):
    by.setdefault(y,[]).append(idx)
labels=sorted(by); rng=np.random.default_rng(0)
tr_lab=labels[:int(0.8*len(labels))]; te_lab=labels[int(0.8*len(labels)):]
def load(idx):
    img,_=ds[idx]; return img
def episode(pool, N=60, Ns=5, Nq=5):
    cls=rng.choice(pool, N, replace=False); S=[]; Q=[]; yq=[]
    for j,c in enumerate(cls):
        pick=rng.choice(by[c], Ns+Nq, replace=False)
        S.append(torch.stack([load(i) for i in pick[:Ns]]))
        Q.append(torch.stack([load(i) for i in pick[Ns:]])); yq+=[j]*Nq
    return torch.stack(S).to(DEV), torch.stack(Q).to(DEV).reshape(-1,1,28,28), torch.tensor(yq).to(DEV)

enc=Encoder().to(DEV); opt=torch.optim.Adam(enc.parameters(),1e-3)
TAU=1.0  # ProtoNet uses squared-euclidean; tau=1 (standard). sigma^2 = tau/2.
print("training ProtoNet on Omniglot...")
for it in range(400):
    enc.train(); S,Q,yq=episode(tr_lab)
    N,Ns=S.shape[:2]; z=enc(S.reshape(-1,1,28,28)).reshape(N,Ns,-1)
    proto=z.mean(1); zq=enc(Q)
    d2=((zq[:,None,:]-proto[None,:,:])**2).sum(-1)   # [Nq*N, N]
    logits=-d2/TAU; loss=F.cross_entropy(logits,yq)
    opt.zero_grad(); loss.backward(); opt.step()
    if it%150==0:
        acc=(logits.argmax(1)==yq).float().mean().item(); print(f" it{it} loss{loss.item():.3f} acc{acc:.3f}")

# ---- certify on held-out episodes ----
enc.eval()
def cert(A,sig,d): return A*np.exp(-d**2/(2*sig*sig))
tail_star=[]; tail_4star=[]; normfar_star=[]; normfar_4star=[]; ident=[]; accs=[]
with torch.no_grad():
    for _ in range(40):
        S,Q,yq=episode(te_lab, N=60, Ns=5, Nq=5)
        N,Ns=S.shape[:2]; z=enc(S.reshape(-1,1,28,28)).reshape(N,Ns,-1)
        proto=z.mean(1).cpu().numpy(); zq=enc(Q).cpu().numpy()   # proto:[N,64], zq:[Nq*N,64]
        d2=((zq[:,None,:]-proto[None,:,:])**2).sum(-1)           # [Q,N]
        logits=-d2/TAU
        accs.append((logits.argmax(1)==yq.cpu().numpy()).mean())
        # identity: protonet softmax == gaussian kernel field over prototypes, v_k=1, sigma^2=tau/2
        sig2=TAU/2.0
        pn=np.exp(logits-logits.max(1,keepdims=True)); pn/=pn.sum(1,keepdims=True)
        kf=np.exp(-d2/(2*sig2)); kf/=kf.sum(1,keepdims=True)
        ident.append(np.abs(pn-kf).max())
        # per query: d = 10th pctile query-prototype distance; near/far split at d
        dist=np.sqrt(d2)                                          # [Q,N]
        for qi in range(dist.shape[0]):
            dd=dist[qi]; dstar=np.percentile(dd,10)
            far=dd>dstar
            if far.sum()<2: continue
            # A measured at the pairwise/single-source scale sigma_pair (mirrors attention_cert.py):
            sig_pair=dstar/np.sqrt(2*np.log(1.0/EPS))
            wfar=np.exp(-(dd[far]**2)/(2*sig_pair**2))            # far weights at sigma_pair, v_k=1
            Neff=(wfar.sum()**2)/(wfar**2).sum()
            Vmax=1.0                                              # BOUNDED weights
            A=Vmax*Neff
            sstar=dstar/np.sqrt(2*np.log(A/EPS)) if A>EPS else dstar
            for scale,TS,NF in [(1.0,tail_star,normfar_star),(4.0,tail_4star,normfar_4star)]:
                s=sstar*scale
                tail=(np.exp(-(dd[far]**2)/(2*s*s))).sum()       # far interference mass, v_k=1
                c=cert(A,s,dstar)
                TS.append(tail/Vmax/EPS)  # certified far tail in eps units; budget=1  # tail in eps units vs budget
                # normalized far mass = far kernel mass / total kernel mass (v_k=1)
                allw=np.exp(-(dd**2)/(2*s*s)); NF.append(allw[far].sum()/allw.sum())
def stat(x): x=np.array(x); return dict(mean=float(x.mean()),viol=int((x>1).sum()),n=len(x))
out=dict(EPS=EPS, tau=TAU, model="protonet-omniglot-conv4-64d", accuracy=float(np.mean(accs)),
    identity_max=float(np.max(ident)),
    tail_star=stat(tail_star), tail_4star=stat(tail_4star),
    normfar_star_mean=float(np.mean(normfar_star)), normfar_4star_mean=float(np.mean(normfar_4star)),
    n_queries=len(tail_star))
json.dump(out, open("proto_cert.json","w"), indent=1)
print("IDENTITY max |protonet-kernelfield|:", out["identity_max"])
print("test-episode acc:", round(out["accuracy"],3))
print("tail/eps at sigma*:", out["tail_star"], "| at 4sigma*:", out["tail_4star"])
print("normalized far mass sigma*:", round(out["normfar_star_mean"],3), "| 4sigma*:", round(out["normfar_4star_mean"],3))
