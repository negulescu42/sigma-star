
import numpy as np, json, time, sys, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
t0=time.time(); dev="cuda" if torch.cuda.is_available() else "cpu"
MODEL=sys.argv[1] if len(sys.argv)>1 else "gpt2-large"
LAYER=int(sys.argv[2]) if len(sys.argv)>2 else 20   # GRACE edits a mid/late MLP layer
N_EDIT=int(sys.argv[3]) if len(sys.argv)>3 else 300
EPS=0.05
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32).to(dev).eval()
nl=model.config.n_layer if hasattr(model.config,"n_layer") else model.config.num_hidden_layers
LAYER=min(LAYER,nl-1)
ds=load_dataset("azhx/counterfact",split="train",streaming=True)
# GRACE key = hidden state at the LAST token of the prompt, at the chosen layer (the edit site).
@torch.no_grad()
def key(prompt):
    ids=tok(prompt,return_tensors="pt",truncation=True,max_length=64).to(dev)
    out=model(**ids,output_hidden_states=True)
    h=out.hidden_states[LAYER][0]   # (T, D)
    return h[-1].float().cpu().numpy()   # last-token activation
edit_keys=[]; para_keys=[]; neigh_keys=[]
it=iter(ds); n=0
for ex in it:
    if n>=N_EDIT: break
    rw=ex["requested_rewrite"]
    # edit prompt: subject filled into the rewrite prompt template
    ep=rw["prompt"].format(rw["subject"]) if "{}" in rw["prompt"] else rw["prompt"]+" "+rw["subject"]
    paras=ex["paraphrase_prompts"][:2]; neighs=ex["neighborhood_prompts"][:2]
    if not paras or not neighs: continue
    try:
        ek=key(ep)
        pk=[key(p) for p in paras]; nk=[key(p) for p in neighs]
    except Exception: continue
    edit_keys.append(ek); para_keys.append(pk); neigh_keys.append(nk); n+=1
edit_keys=np.array(edit_keys)                    # (N, D)
# distances in key space (GRACE uses L2 in activation space)
def dists(edit, group):
    out=[]
    for i,ek in enumerate(edit):
        for gk in group[i]:
            out.append(np.linalg.norm(ek-gk))
    return np.array(out)
d_para=dists(edit_keys,para_keys)     # in-scope: should be INSIDE radius (activate edit)
d_neigh=dists(edit_keys,neigh_keys)   # out-of-scope: should be OUTSIDE radius (defer)
# calibrate sigma* as a deferral radius from the edit-key field geometry (label-free):
# d = 10th pct of pairwise EDIT-key distances (the near/far separation among stored edits);
# A = participation ratio of far kernel weights; sigma* = d/sqrt(2 ln(A/eps)).
P=np.linalg.norm(edit_keys[:,None,:]-edit_keys[None,:,:],axis=2)
pv=P[P>0]; dsep=float(np.percentile(pv,10))
def Ksig(t2,s): return np.exp(-t2/(2*s*s))
sig_pair=dsep/np.sqrt(2*np.log(1/EPS))
# A at sig_pair over edit keys as queries, far set = other edit keys beyond dsep
om_all=[]
for i in range(len(edit_keys)):
    r=P[i]; far=r>dsep; w=Ksig(r[far]**2,sig_pair)
    if w.sum()>0: om_all.append((w.sum()**2)/(w**2).sum())
A=float(np.mean(om_all)); sig_star=dsep/np.sqrt(2*np.log(A/EPS))
# The GRACE radius is a threshold on key distance. We report separation of para vs neigh,
# and where sigma*-derived radius sits. A natural GRACE radius ~ a few sigma (kernel support).
def sep_at(radius):
    tpr=(d_para<=radius).mean()   # fraction of paraphrases correctly activated (in-scope)
    fpr=(d_neigh<=radius).mean()  # fraction of neighbours wrongly activated (should defer)
    return tpr,fpr
# ROC over radius; AUROC of key-distance separating para(in) from neigh(out)
from numpy import trapz
rads=np.linspace(0, max(d_para.max(),d_neigh.max()), 400)
tprs=np.array([(d_para<=r).mean() for r in rads]); fprs=np.array([(d_neigh<=r).mean() for r in rads])
order=np.argsort(fprs); auroc=float(trapz(tprs[order],fprs[order]))
# candidate GRACE radii tied to sigma*: r = c*sigma* for c in {1,2,3}
cand={}
for c in [1.0,2.0,3.0]:
    tpr,fpr=sep_at(c*sig_star); cand[f"{c}sigma*"]={"radius":c*sig_star,"para_TPR":float(tpr),"neigh_FPR":float(fpr)}
res={"model":MODEL,"layer":LAYER,"n_edit":len(edit_keys),"dim":int(edit_keys.shape[1]),
     "d_sep_10pct":dsep,"A":A,"sigma_star":float(sig_star),
     "d_para":{"mean":float(d_para.mean()),"median":float(np.median(d_para))},
     "d_neigh":{"mean":float(d_neigh.mean()),"median":float(np.median(d_neigh))},
     "auroc_key_distance":auroc,"radius_candidates":cand}
print(json.dumps(res,indent=2))
json.dump(res,open("grace_defer.json","w"),indent=2)
print(f"WROTE grace_defer.json {time.time()-t0:.1f}s")
