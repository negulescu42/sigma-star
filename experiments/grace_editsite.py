
import numpy as np, json, time, sys, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from numpy import trapz
t0=time.time(); dev="cuda" if torch.cuda.is_available() else "cpu"
MODEL="gpt2-large"; N_EDIT=250
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32).to(dev).eval()
nl=model.config.n_layer
# GRACE edits the input to the MLP down-projection (c_proj) of a chosen block.
# Hook the input to block.mlp.c_proj at layers {18,24,30}, last token.
EDIT_LAYERS=[18,24,30]
hooks={}; captured={}
def mk(L):
    def hook(mod,inp,out): captured[L]=inp[0].detach()   # (B,T,4d) input to c_proj
    return hook
for L in EDIT_LAYERS:
    blk=model.transformer.h[L].mlp.c_proj
    hooks[L]=blk.register_forward_hook(mk(L))
ds=load_dataset("azhx/counterfact",split="train",streaming=True)
@torch.no_grad()
def editkeys(prompt):
    ids=tok(prompt,return_tensors="pt",truncation=True,max_length=64).to(dev)
    _=model(**ids); res={}
    for L in EDIT_LAYERS: res[L]=captured[L][0,-1].float().cpu().numpy()
    return res
E={L:[] for L in EDIT_LAYERS}; P={L:[] for L in EDIT_LAYERS}; Nn={L:[] for L in EDIT_LAYERS}
cnt=0
for ex in iter(ds):
    if cnt>=N_EDIT: break
    rw=ex["requested_rewrite"]
    ep=rw["prompt"].format(rw["subject"]) if "{}" in rw["prompt"] else rw["prompt"]+" "+rw["subject"]
    paras=ex["paraphrase_prompts"][:1]; neighs=ex["neighborhood_prompts"][:1]
    if not paras or not neighs: continue
    try: ek=editkeys(ep); pk=editkeys(paras[0]); nk=editkeys(neighs[0])
    except Exception: continue
    for L in EDIT_LAYERS: E[L].append(ek[L]); P[L].append(pk[L]); Nn[L].append(nk[L])
    cnt+=1
def auroc(edit,para,neigh,W=None):
    edit=np.array(edit); para=np.array(para); neigh=np.array(neigh)
    if W is not None: edit=edit@W; para=para@W; neigh=neigh@W
    dp=np.linalg.norm(edit-para,axis=1); dn=np.linalg.norm(edit-neigh,axis=1)
    rads=np.linspace(0,max(dp.max(),dn.max()),300)
    tpr=np.array([(dp<=r).mean() for r in rads]); fpr=np.array([(dn<=r).mean() for r in rads])
    o=np.argsort(fpr); return float(trapz(tpr[o],fpr[o]))
out={"model":MODEL,"n_edit":cnt,"editsite":{}}
for L in EDIT_LAYERS:
    raw=auroc(E[L],P[L],Nn[L])
    # whitening transform from edit keys
    X=np.array(E[L]); mu=X.mean(0); Xc=X-mu
    C=np.cov(Xc.T)+1e-3*np.eye(X.shape[1]); vals,vecs=np.linalg.eigh(C)
    W=vecs@np.diag(1.0/np.sqrt(np.clip(vals,1e-6,None)))@vecs.T
    Ew=[e-mu for e in E[L]]; Pw=[p-mu for p in P[L]]; Nw=[n-mu for n in Nn[L]]
    wh=auroc(Ew,Pw,Nw,W)
    out["editsite"][f"cproj_in_L{L}"]={"auroc_raw":raw,"auroc_whitened":wh}
    print(f"cproj_in_L{L}: raw AUROC={raw:.3f}  whitened AUROC={wh:.3f}")
best=max([(k,max(v['auroc_raw'],v['auroc_whitened'])) for k,v in out['editsite'].items()],key=lambda x:x[1])
out["best"]=best; print("BEST:",best)
json.dump(out,open("grace_editsite.json","w"),indent=2)
print(f"WROTE {time.time()-t0:.1f}s")
