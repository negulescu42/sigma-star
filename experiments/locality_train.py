"""
Locality-certified by construction (control experiment). FINAL STATE = negative control.

Tests the RCP conclusion's future-work proposal -- make the differentiable far-tail a
training objective so a field is "locality-certified by construction" -- and, per the
R36 referee, isolates the causal claim with a matched generic-regularization control
and multi-seed s.d. The verdict is a NULL: training against the tail is redundant
because sigma* is by construction the bandwidth where the field already holds the budget.

Design:
  * Representation head + STANDARD linear softmax classifier (clf = nn.Linear(D_OUT, NCLS)),
    which carries NO built-in locality pressure (an earlier kernel-vote baseline was
    itself local and confounded the test). project() standardizes each output dim to unit
    variance, so no arm can lower the tail by globally shrinking coordinates.
  * A seed-0 baseline fixes the aggregate operating geometry (radius d, mass A0, certified
    sigma*). The field is then DEPLOYED and penalized at an OVER-WIDE bandwidth
    sig_op = MULT * sigma* (MULT = 4.0), the coarse regime where a plain field is over
    budget -- the only regime where the objective could be non-trivial. All arms share
    this single (d, sig_op) frame (no geometry-rescaling confound).
  * Arms (5 seeds each; report mean +/- s.d. for acc, tail/eps, A, cert-local frac):
      plain: CE softmax only                     (ordinary training)
      tail : CE + lambda * Tail_{sig_op}, lam=0.3 (the certificate as objective)
      l2   : CE + weight_decay=1e-3               (generic-regularizer control)
  * Finding: at sigma* all three arms are identical (already 100% certified-local); at
    4*sigma* all are far over budget and the tail penalty makes the tail slightly WORSE
    while costing ~13 accuracy points. Writes keys: seeds, d_dep, sigma_dep, sigma_star,
    mult, A0, arms, results{plain,tail,l2}. (An earlier single-seed lambda-sweep showing
    a spurious 39x/+1.8pt "win" was a bandwidth-mismatch artifact, since retracted.)
"""
import numpy as np, torch, torch.nn as nn, json, time
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 0.05; FEAT = "/workspace/cifar_feats.npz"; SEEDS = [0,1,2,3,4]

z = np.load(FEAT)
Xtr, ytr = z["Xtr"].astype(np.float32), z["Ytr"].astype(np.int64)
Xte, yte = z["Xte"].astype(np.float32), z["Yte"].astype(np.int64)
mu, sd = Xtr.mean(0,keepdims=True), Xtr.std(0,keepdims=True)+1e-6
Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
Xtr_t=torch.tensor(Xtr,device=DEV); ytr_t=torch.tensor(ytr,device=DEV)
Xte_t=torch.tensor(Xte,device=DEV); yte_t=torch.tensor(yte,device=DEV)
D_IN=Xtr.shape[1]; NCLS=int(ytr.max())+1
print(f"train {Xtr.shape} test {Xte.shape} nclass {NCLS}", flush=True)

def balanced_idx(y, per_class, seed=0):
    rng=np.random.default_rng(seed); out=[]
    for c in range(NCLS):
        ci=np.where(y==c)[0]; out.append(rng.choice(ci,size=min(per_class,len(ci)),replace=False))
    return np.concatenate(out)
fld=balanced_idx(ytr,80); Xf_t=Xtr_t[fld]; yf_t=ytr_t[fld]

def project(head,X):
    h=head(X); return (h-h.mean(0,keepdim=True))/(h.std(0,keepdim=True)+1e-6)
def pdist2(a,b): return (a*a).sum(1,keepdim=True)+(b*b).sum(1)-2*a@b.T
@torch.no_grad()
def tenth_pct_radius(P):
    n=min(2000,P.shape[0]); s=P[torch.randperm(P.shape[0],device=P.device)[:n]]
    d=torch.sqrt(torch.clamp(pdist2(s,s),min=0)); return torch.quantile(d[d>0].flatten()[:200000],0.10)
def class_scores(Pq,Pf,yf,sig):
    K=torch.exp(-pdist2(Pq,Pf)/(2*sig**2)); S=torch.zeros(Pq.shape[0],NCLS,device=Pq.device)
    S.index_add_(1,yf,K); return S

def train_head(mode, strength, d_dep, sig_dep, seed, steps=600, B=256, D_OUT=128, lr=1e-3):
    """Representation head + STANDARD linear softmax classifier (no locality pressure).
    mode='tail' adds lambda*Tail_{sigma*}; mode='l2' adds weight decay on the head;
    mode='plain' is the ordinary-training baseline."""
    torch.manual_seed(seed)
    head=nn.Linear(D_IN,D_OUT).to(DEV)
    clf=nn.Linear(D_OUT,NCLS).to(DEV)
    params=list(head.parameters())+list(clf.parameters())
    opt=torch.optim.Adam(params, lr=lr, weight_decay=(strength if mode=="l2" else 0.0))
    d2dep=torch.tensor(d_dep**2,device=DEV); sg=torch.tensor(sig_dep,device=DEV)
    for t in range(steps):
        bi=torch.randint(0,Xtr_t.shape[0],(B,),device=DEV)
        P=project(head,Xtr_t[bi])
        logits=clf(P)                                   # standard softmax classifier
        loss=nn.functional.cross_entropy(logits, ytr_t[bi])
        if mode=="tail":
            Pf=project(head,Xf_t); d2=pdist2(P,Pf); K=torch.exp(-d2/(2*sg**2))
            loss=loss+strength*(K*(d2>d2dep)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return head

@torch.no_grad()
def evaluate(head,d_dep,sig_dep):
    Pf=project(head,Xf_t); Pte=project(head,Xte_t); d2dep=d_dep**2; B=1000
    A=[]; tail=[]; correct=0; ntot=0
    for s in range(0,Pte.shape[0],B):
        Pq=Pte[s:s+B]; d2=pdist2(Pq,Pf); K=torch.exp(-d2/(2*sig_dep**2)); w=K*(d2>d2dep)
        A.append(((w.sum(1)**2)/((w*w).sum(1)+1e-30)).cpu()); tail.append(w.sum(1).cpu())
        S=class_scores(Pq,Pf,yf_t,torch.tensor(sig_dep,device=DEV))
        correct+=(S.argmax(1)==yte_t[s:s+B]).sum().item(); ntot+=Pq.shape[0]
    A=torch.cat(A); tail=torch.cat(tail)
    return dict(acc=correct/ntot, A_mean=A.mean().item(),
                tail_over_eps=tail.mean().item()/EPS, cert_local_frac=(tail<=EPS).float().mean().item())

t0=time.time()
# fixed deployment geometry from a seed-0 baseline
base0=train_head("plain",0.0,1.0,1.0,seed=0)
Pf0=project(base0,Xf_t); d0=tenth_pct_radius(Pf0).item()
sig_ss=d0/np.sqrt(2*np.log(1/EPS))
@torch.no_grad()
def agg_A(head,d,sig):
    Pf=project(head,Xf_t); Pte=project(head,Xte_t); d2dep=d**2; B=1000; A=[]
    for s in range(0,Pte.shape[0],B):
        Pq=Pte[s:s+B]; d2=pdist2(Pq,Pf); K=torch.exp(-d2/(2*sig**2)); w=K*(d2>d2dep)
        A.append(((w.sum(1)**2)/((w*w).sum(1)+1e-30)).cpu())
    return torch.cat(A).mean().item()
A0=agg_A(base0,d0,sig_ss); sig_star=d0/np.sqrt(2*np.log(max(A0,1.0001)/EPS))
MULT=4.0                      # operate at 4x the certified bandwidth (over-wide)
sig_dep=MULT*sig_star
print(f"deployment: d0={d0:.3f} A0={A0:.1f} sigma*={sig_star:.4f} MULT={MULT} sig_op={sig_dep:.4f}", flush=True)

# arms: baseline (lambda=0), TAIL (lambda=0.03), L2 (wd matched to give a comparable accuracy band)
ARMS = {"plain":("plain",0.0), "tail":("tail",0.3), "l2":("l2",1e-3)}
def run_arm(mode,strength):
    evs=[evaluate(train_head(mode,strength,d0,sig_dep,seed=s),d0,sig_dep) for s in SEEDS]
    keys=evs[0].keys()
    return {k:{"mean":float(np.mean([e[k] for e in evs])),"sd":float(np.std([e[k] for e in evs]))} for k in keys}
results={name:run_arm(m,s) for name,(m,s) in ARMS.items()}
for name in ARMS:
    r=results[name]
    print(f"{name:>9}: acc={r['acc']['mean']*100:.1f}+/-{r['acc']['sd']*100:.1f}%  "
          f"tail/eps={r['tail_over_eps']['mean']:.4f}  A={r['A_mean']['mean']:.1f}  "
          f"cert_local={r['cert_local_frac']['mean']*100:.1f}%", flush=True)
summary=dict(eps=EPS,d_in=D_IN,nclass=NCLS,field_points=int(len(fld)),seeds=SEEDS,
             d_dep=d0,sigma_dep=sig_dep,sigma_star=sig_star,mult=MULT,A0=A0,arms={k:list(v) for k,v in ARMS.items()},
             results=results,wall_sec=round(time.time()-t0,1))
json.dump(summary,open("locality_train.json","w"),indent=2)
print("WROTE locality_train.json", flush=True)
