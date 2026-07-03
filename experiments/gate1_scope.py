"""GATE 1 (RCP upgrade plan): does the near/far separation (Assumption A2) that
FAILS in raw LLM key space recover in a LEARNED / semantic scope geometry, on
HELD-OUT edits? Reproduces the SS6 raw baseline (~0.60), then tests two routes.
Route A: contrastive low-rank projection phi learned on TRAIN edits, frozen, eval on HELD-OUT.
Route B: frozen MPNet sentence encoder as scope space (no per-edit training).
Go/no-go: held-out near/far AUROC >= ~0.85 = GO. seed=0."""
import numpy as np, json, time, torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
t0=time.time(); dev="cuda" if torch.cuda.is_available() else "cpu"
rng=np.random.default_rng(0); torch.manual_seed(0)
MODEL="gpt2-large"; N_EDIT=1500; EDIT_LAYERS=[18,24,30]; PARA_K=2; NEIGH_K=4
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32).to(dev).eval()
captured={}
def mk(L):
    def hook(mod,inp,out): captured[L]=inp[0].detach()
    return hook
for L in EDIT_LAYERS: model.transformer.h[L].mlp.c_proj.register_forward_hook(mk(L))
@torch.no_grad()
def keys(prompt):
    ids=tok(prompt,return_tensors="pt",truncation=True,max_length=64).to(dev); _=model(**ids)
    return {L:captured[L][0,-1].float().cpu().numpy() for L in EDIT_LAYERS}

ds=load_dataset("azhx/counterfact",split="train",streaming=True)
edits=[]  # each: {ekey[L], para[L] list, neigh[L] list, text fields}
cnt=0
for ex in iter(ds):
    if cnt>=N_EDIT: break
    rw=ex["requested_rewrite"]
    ep=rw["prompt"].format(rw["subject"]) if "{}" in rw["prompt"] else rw["prompt"]+" "+rw["subject"]
    paras=ex["paraphrase_prompts"][:PARA_K]; neighs=ex["neighborhood_prompts"][:NEIGH_K]
    if len(paras)<1 or len(neighs)<1: continue
    try:
        ek=keys(ep); pk=[keys(p) for p in paras]; nk=[keys(n) for n in neighs]
    except Exception: continue
    edits.append({"ek":ek,"pk":pk,"nk":nk,"t_e":ep,"t_p":paras,"t_n":neighs}); cnt+=1
n=len(edits); print(f"edits={n}  ({time.time()-t0:.0f}s)",flush=True)
idx=rng.permutation(n); tr=idx[:int(0.67*n)]; te=idx[int(0.67*n):]

def auroc_pairs(dp,dn):  # dp: near dists (should be small), dn: far dists
    dp=np.asarray(dp); dn=np.asarray(dn); 
    lab=np.r_[np.ones_like(dp),np.zeros_like(dn)]; sc=-np.r_[dp,dn]  # smaller dist -> more "near"
    o=np.argsort(-sc); lab=lab[o]; P=lab.sum(); N=len(lab)-P
    if P==0 or N==0: return float("nan")
    tp=np.cumsum(lab); fp=np.cumsum(1-lab)
    return float(np.trapz(tp/P, fp/N))

def eval_raw(subset,L,W=None,mu=None):
    dp=[]; dn=[]
    for i in subset:
        e=edits[i]["ek"][L].copy(); 
        if mu is not None: e=e-mu
        if W is not None: e=e@W
        for p in edits[i]["pk"]:
            pp=p[L].copy()
            if mu is not None: pp=pp-mu
            if W is not None: pp=pp@W
            dp.append(np.linalg.norm(e-pp))
        for nn_ in edits[i]["nk"]:
            npv=nn_[L].copy()
            if mu is not None: npv=npv-mu
            if W is not None: npv=npv@W
            dn.append(np.linalg.norm(e-npv))
    return auroc_pairs(dp,dn)

results={"model":MODEL,"n_edit":n,"n_train":len(tr),"n_heldout":len(te),"layers":{}}
# ---- raw + whitened baselines (reproduce SS6) ----
for L in EDIT_LAYERS:
    raw_te=eval_raw(te,L)
    X=np.array([edits[i]["ek"][L] for i in tr]); mu=X.mean(0); Xc=X-mu
    C=np.cov(Xc.T)+1e-3*np.eye(X.shape[1]); vals,vecs=np.linalg.eigh(C)
    W=(vecs@np.diag(1/np.sqrt(np.clip(vals,1e-6,None)))@vecs.T).astype("float32")
    wh_te=eval_raw(te,L,W=W,mu=mu)
    results["layers"][f"L{L}"]={"auroc_raw_heldout":raw_te,"auroc_whitened_heldout":wh_te}
    print(f"[L{L}] raw held-out AUROC={raw_te:.3f}  whitened={wh_te:.3f}",flush=True)

# ---- Route A: contrastive low-rank phi trained on TRAIN edits ----
def build_XYZ(subset,L):
    E=[];Pp=[];Nn=[]
    for i in subset:
        E.append(edits[i]["ek"][L]); Pp.append([p[L] for p in edits[i]["pk"]]); Nn.append([nn_[L] for nn_ in edits[i]["nk"]])
    return E,Pp,Nn
bestL=max(results["layers"],key=lambda k:results["layers"][k]["auroc_raw_heldout"])
Lb=int(bestL[1:])
din=edits[0]["ek"][Lb].shape[0]
class Phi(nn.Module):
    def __init__(s,din,d=128): super().__init__(); s.W=nn.Linear(din,d,bias=False)
    def forward(s,x): z=s.W(x); return z/ (z.norm(dim=-1,keepdim=True)+1e-8)
phi=Phi(din).to(dev); opt=torch.optim.Adam(phi.parameters(),lr=1e-3,weight_decay=1e-4)
Etr,Ptr,Ntr=build_XYZ(tr,Lb)
Et=torch.tensor(np.array(Etr),dtype=torch.float32,device=dev)
# stack paraphrase/neighbor tensors (ragged -> pad by sampling)
def sample_triples(bs=256):
    ii=rng.integers(0,len(Etr),bs)
    e=Et[ii]
    p=torch.tensor(np.array([Ptr[i][rng.integers(0,len(Ptr[i]))] for i in ii]),dtype=torch.float32,device=dev)
    ng=torch.tensor(np.array([Ntr[i][rng.integers(0,len(Ntr[i]))] for i in ii]),dtype=torch.float32,device=dev)
    return e,p,ng
margin=0.2
for step in range(1500):
    e,p,ng=sample_triples()
    ze,zp,zn=phi(e),phi(p),phi(ng)
    dpos=1-(ze*zp).sum(-1); dneg=1-(ze*zn).sum(-1)
    loss=torch.clamp(dpos-dneg+margin,min=0).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step%500==0: print(f"  routeA step {step} loss {loss.item():.4f}",flush=True)
phi.eval()
@torch.no_grad()
def eval_phi(subset):
    dp=[];dn=[]
    for i in subset:
        e=phi(torch.tensor(edits[i]["ek"][Lb],dtype=torch.float32,device=dev)[None])[0].cpu().numpy()
        for p in edits[i]["pk"]:
            zp=phi(torch.tensor(p[Lb],dtype=torch.float32,device=dev)[None])[0].cpu().numpy(); dp.append(np.linalg.norm(e-zp))
        for nn_ in edits[i]["nk"]:
            zn=phi(torch.tensor(nn_[Lb],dtype=torch.float32,device=dev)[None])[0].cpu().numpy(); dn.append(np.linalg.norm(e-zn))
    return auroc_pairs(dp,dn)
routeA_tr=eval_phi(tr); routeA_te=eval_phi(te)
results["routeA_contrastive"]={"layer":Lb,"auroc_train":routeA_tr,"auroc_heldout":routeA_te}
print(f"[Route A phi @L{Lb}] train AUROC={routeA_tr:.3f}  HELD-OUT AUROC={routeA_te:.3f}",flush=True)

# ---- Route B: frozen MPNet sentence encoder as scope space (no training) ----
try:
    from sentence_transformers import SentenceTransformer
    enc=SentenceTransformer("all-mpnet-base-v2",device=dev)
    def embed(txts): return enc.encode(txts,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    dp=[];dn=[]
    for i in te:
        e=embed([edits[i]["t_e"]])[0]
        pe=embed(edits[i]["t_p"]); ne=embed(edits[i]["t_n"])
        for v in pe: dp.append(np.linalg.norm(e-v))
        for v in ne: dn.append(np.linalg.norm(e-v))
    routeB_te=auroc_pairs(dp,dn)
    results["routeB_mpnet_frozen"]={"auroc_heldout":routeB_te,"trained":False}
    print(f"[Route B MPNet frozen] HELD-OUT AUROC={routeB_te:.3f}",flush=True)
except Exception as ex:
    results["routeB_mpnet_frozen"]={"error":str(ex)}
    print("Route B error:",ex,flush=True)

# ---- gate verdict ----
cands={"routeA":results["routeA_contrastive"]["auroc_heldout"]}
if "auroc_heldout" in results.get("routeB_mpnet_frozen",{}): cands["routeB"]=results["routeB_mpnet_frozen"]["auroc_heldout"]
best_route=max(cands,key=cands.get); best_auroc=cands[best_route]
verdict="GO" if best_auroc>=0.85 else ("PARTIAL" if best_auroc>=0.65 else "NO-GO")
results["gate1"]={"best_route":best_route,"best_heldout_auroc":best_auroc,"verdict":verdict,
                  "raw_baseline_heldout":results["layers"][bestL]["auroc_raw_heldout"]}
json.dump(results,open("gate1_scope.json","w"),indent=2)
print(f"\nGATE 1 VERDICT: {verdict}  (best {best_route} held-out AUROC={best_auroc:.3f} vs raw {results['layers'][bestL]['auroc_raw_heldout']:.3f})")
print(f"WROTE gate1_scope.json  {time.time()-t0:.0f}s")
