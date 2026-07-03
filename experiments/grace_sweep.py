
import numpy as np, json, time, sys, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from numpy import trapz
t0=time.time(); dev="cuda" if torch.cuda.is_available() else "cpu"
MODEL="gpt2-large"; N_EDIT=250
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32).to(dev).eval()
nl=model.config.n_layer
ds=load_dataset("azhx/counterfact",split="train",streaming=True)
LAYERS=[6,12,18,24,30,nl-1]; POS=["last","mean"]
@torch.no_grad()
def keys_all(prompt):
    ids=tok(prompt,return_tensors="pt",truncation=True,max_length=64).to(dev)
    out=model(**ids,output_hidden_states=True)
    res={}
    for L in LAYERS:
        h=out.hidden_states[L][0]
        res[("last",L)]=h[-1].float().cpu().numpy()
        res[("mean",L)]=h.mean(0).float().cpu().numpy()
    return res
E={}; P={}; Nn={}
for k in [(p,L) for p in POS for L in LAYERS]: E[k]=[]; P[k]=[]; Nn[k]=[]
cnt=0
for ex in iter(ds):
    if cnt>=N_EDIT: break
    rw=ex["requested_rewrite"]
    ep=rw["prompt"].format(rw["subject"]) if "{}" in rw["prompt"] else rw["prompt"]+" "+rw["subject"]
    paras=ex["paraphrase_prompts"][:1]; neighs=ex["neighborhood_prompts"][:1]
    if not paras or not neighs: continue
    try:
        ek=keys_all(ep); pk=keys_all(paras[0]); nk=keys_all(neighs[0])
    except Exception: continue
    for k in E: E[k].append(ek[k]); P[k].append(pk[k]); Nn[k].append(nk[k])
    cnt+=1
def auroc(edit,para,neigh):
    edit=np.array(edit); dp=np.array([np.linalg.norm(edit[i]-para[i]) for i in range(len(edit))])
    dn=np.array([np.linalg.norm(edit[i]-neigh[i]) for i in range(len(edit))])
    rads=np.linspace(0,max(dp.max(),dn.max()),300)
    tpr=np.array([(dp<=r).mean() for r in rads]); fpr=np.array([(dn<=r).mean() for r in rads])
    o=np.argsort(fpr); return float(trapz(tpr[o],fpr[o])), float(dp.mean()), float(dn.mean())
out={"model":MODEL,"n_edit":cnt,"sweep":{}}
best=None
for k in E:
    a,mp,mn=auroc(E[k],P[k],Nn[k]); tag=f"{k[0]}_L{k[1]}"
    out["sweep"][tag]={"auroc":a,"d_para":mp,"d_neigh":mn}
    print(f"{tag}: AUROC={a:.3f} d_para={mp:.1f} d_neigh={mn:.1f}")
    if best is None or a>best[1]: best=(tag,a)
out["best"]=best
print("BEST:",best)
json.dump(out,open("grace_sweep.json","w"),indent=2)
print(f"WROTE grace_sweep.json {time.time()-t0:.1f}s")
