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
for it in range(250):
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
# ONE-SHOT GIBBS CERTIFICATE (corrected). Reference temperature tau0 = deployment TAU.
# Measure D_{q,0}=D_q(tau0); one-shot tau_OS = Delta/ln(D_{q,0}/eps); DEPLOY at
# tau_cert = min(tau0, tau_OS). Theorem: m_F(tau_cert) <= eps for every query:
#   - if tau_OS <= tau0 (cooling): D_q(tau_cert)<=D_{q,0} (flattening) so m_F<=D_{q,0}e^{-Delta/tau_cert}=eps
#   - if tau_OS >  tau0 (already certified at tau0): bound(tau0)=D_{q,0}e^{-Delta/tau0}<eps so m_F(tau0)<eps
TAU0=TAU
agg={q:{"bound0":[],"mdep":[],"viol_dep":0,"mcert":[],"viol_cert":0,
        "tau_cert":[],"need_cool":0,"n":0} for q in QS}
accs=[]
def massat(E,tau,far):
    z=(-E)/tau; z=z-z.max(); a=np.exp(z); a/=a.sum(); return float(a[far].sum())
with torch.no_grad():
    for _ in range(24):
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
            m_dep=massat(E,TAU0,far)                  # deployment far mass at tau0
            wf0=np.exp(-(E[far]-r_near2)/TAU0)        # far Gibbs weights at reference tau0
            for q in QS:
                Dq0=hill(wf0,q); bound0=float(Dq0*np.exp(-Delta/TAU0))
                tau_os=Delta/np.log(max(Dq0,1.001)/EPS)
                tau_cert=min(TAU0,tau_os)             # <-- the correction
                m_cert=massat(E,tau_cert,far)
                a=agg[q]
                a["bound0"].append(bound0); a["mdep"].append(m_dep)
                a["viol_dep"]+=int(m_dep>bound0+1e-9)
                a["mcert"].append(m_cert); a["viol_cert"]+=int(m_cert>EPS+1e-6)
                a["tau_cert"].append(tau_cert); a["need_cool"]+=int(tau_os<TAU0); a["n"]+=1
out={"EPS":EPS,"tau0":TAU0,"model":"protonet-omniglot-conv4-64d","accuracy":float(np.mean(accs)),
     "certificate":"one-shot min(tau0,tau_OS)"}
for q in QS:
    a=agg[q]; key="inf" if np.isinf(q) else int(q)
    b0=np.array(a["bound0"]); md=np.array(a["mdep"]); mc=np.array(a["mcert"]); tc=np.array(a["tau_cert"])
    out[f"q{key}"]={"n":a["n"],
        "bound0_mean":float(b0.mean()),                       # bound at deployment tau0
        "mdep_mean":float(md.mean()),"bound_violations_at_tau0":int(a["viol_dep"]),
        "mcert_mean":float(mc.mean()),"mcert_max":float(mc.max()),
        "cert_violations":int(a["viol_cert"]),                # target: 0
        "tau_cert_mean":float(tc.mean()),"tau_cert_median":float(np.median(tc)),
        "frac_needing_cooling":float(a["need_cool"]/a["n"])}
json.dump(out,open("gibbs_proto.json","w"),indent=2)
print("acc",round(out["accuracy"],3),"| q2",out["q2"],"| qinf",out["qinf"])
print("SAVED gibbs_proto.json")
