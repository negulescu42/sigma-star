#!/usr/bin/env python3
"""Normalized (Gibbs) RCP certificate on a real ProtoNet.
ProtoNet reads p(k|x)=softmax(-||f(x)-c_k||^2/tau): a Gibbs field with energies
E_k=||f(x)-c_k||^2. Near anchor = nearest prototype (E_0=r_near^2); far set at gap
Delta = d_far^2 - r_near^2. Gibbs bound: m_F <= D_q^{(F)} exp(-Delta/tau);
certified tau* = Delta/ln(D_q/eps) bounds NORMALISED far mass directly (v_k irrelevant).
Trains conv4 ProtoNet on Omniglot, certifies prototype fields on held-out 60-way episodes.
"""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
torch.manual_seed(20260707); np.random.seed(20260707)
EPS=0.05; DEV="cuda" if torch.cuda.is_available() else "cpu"; QS=[2.0,np.inf]
DATA="/workspace/cs_jobs/omniglot"

def conv_block(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2))
class Encoder(nn.Module):
    def __init__(s):
        super().__init__(); s.e=nn.Sequential(conv_block(1,64),conv_block(64,64),conv_block(64,64),conv_block(64,64))
    def forward(s,x): return s.e(x).flatten(1)

tf=transforms.Compose([transforms.Grayscale(),transforms.Resize((28,28)),transforms.ToTensor()])
ds=datasets.Omniglot(root=DATA,background=True,download=True,transform=tf)
by=dict()
for idx,(_,y) in enumerate(ds._flat_character_images): by.setdefault(y,[]).append(idx)
labels=sorted(by); rng=np.random.default_rng(0)
tr_lab=labels[:int(0.8*len(labels))]; te_lab=labels[int(0.8*len(labels)):]
def load(idx): img,_=ds[idx]; return img
def episode(pool,N=60,Ns=5,Nq=5):
    cls=rng.choice(pool,N,replace=False); S=[]; Q=[]; yq=[]
    for j,c in enumerate(cls):
        pick=rng.choice(by[c],Ns+Nq,replace=False)
        S.append(torch.stack([load(i) for i in pick[:Ns]]))
        Q.append(torch.stack([load(i) for i in pick[Ns:]])); yq+=[j]*Nq
    return torch.stack(S).to(DEV),torch.stack(Q).to(DEV).reshape(-1,1,28,28),torch.tensor(yq).to(DEV)

enc=Encoder().to(DEV); opt=torch.optim.Adam(enc.parameters(),1e-3); TAU=1.0
print("training ProtoNet...")
for it in range(400):
    enc.train(); S,Q,yq=episode(tr_lab); N,Ns=S.shape[:2]
    z=enc(S.reshape(-1,1,28,28)).reshape(N,Ns,-1); proto=z.mean(1); zq=enc(Q)
    d2=((zq[:,None,:]-proto[None,:,:])**2).sum(-1); loss=F.cross_entropy(-d2/TAU,yq)
    opt.zero_grad(); loss.backward(); opt.step()
    if it%150==0: print(f" it{it} loss{loss.item():.3f}")

def hill(w,q):
    w=np.clip(w,0,None); s=w.sum()
    if s<=0: return 1.0
    if np.isinf(q): return float(s/(w.max()+1e-30))
    p=w/s; return float((np.power(p,q).sum())**(1.0/(1.0-q)))

enc.eval()
agg={q:{"bound":[],"meas":[],"viol":0,"meas_star":[],"star_viol":0,"n":0} for q in QS}
accs=[]
with torch.no_grad():
    for _ in range(40):
        S,Q,yq=episode(te_lab); N,Ns=S.shape[:2]
        z=enc(S.reshape(-1,1,28,28)).reshape(N,Ns,-1); proto=z.mean(1).cpu().numpy(); zq=enc(Q).cpu().numpy()
        d2=((zq[:,None,:]-proto[None,:,:])**2).sum(-1)   # [Q,N] energies
        accs.append(((-d2).argmax(1)==yq.cpu().numpy()).mean())
        for qi in range(d2.shape[0]):
            E=d2[qi]; order=np.argsort(E)            # nearest prototype first
            r_near2=E[order[0]]; far=order[1:]        # anchor = nearest; far = rest
            if len(far)<2: continue
            Delta=float(E[far].min()-r_near2)
            if Delta<=0: continue
            # deployment normalised far mass at tau=TAU
            z_=(-E)/TAU; z_-=z_.max(); a=np.exp(z_); a/=a.sum(); m_meas=float(a[far].sum())
            wf=np.exp(-(E[far]-r_near2)/TAU)          # far Gibbs weights, anchor-shifted
            for q in QS:
                Dq=hill(wf,q); bound=float(Dq*np.exp(-Delta/TAU))
                agg[q]["bound"].append(bound); agg[q]["meas"].append(m_meas)
                agg[q]["viol"]+=int(m_meas>bound+1e-9); agg[q]["n"]+=1
                taustar=Delta/np.log(max(Dq,1.001)/EPS)
                zs=(-E)/taustar; zs-=zs.max(); asx=np.exp(zs); asx/=asx.sum(); ms=float(asx[far].sum())
                agg[q]["meas_star"].append(ms); agg[q]["star_viol"]+=int(ms>EPS+1e-6)
out={"EPS":EPS,"tau":TAU,"model":"protonet-omniglot-conv4-64d","accuracy":float(np.mean(accs))}
for q in QS:
    b=np.array(agg[q]["bound"]); m=np.array(agg[q]["meas"]); ms=np.array(agg[q]["meas_star"])
    key="inf" if np.isinf(q) else int(q)
    out[f"q{key}"]={"n":agg[q]["n"],"bound_mean":float(b.mean()),"measured_normfar_mean":float(m.mean()),
        "bound_violations":int(agg[q]["viol"]),
        "taustar_meas_normfar_mean":float(ms.mean()),"taustar_over_eps_violations":int(agg[q]["star_viol"])}
json.dump(out,open("gibbs_proto.json","w"),indent=2)
print("acc",round(out["accuracy"],3),"| q2",out["q2"],"| qinf",out["qinf"])
print("SAVED gibbs_proto.json")
