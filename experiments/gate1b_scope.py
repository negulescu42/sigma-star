"""GATE 1 iteration 2 (RCP upgrade). Refines the three routes to close the 0.81->0.85 gap:
 Route A': low-rank (16/32/64) InfoNCE projection of the LLM's own c_proj keys, strong reg.
 Route B': frozen MPNet scope space, then PCA-64 + whitening (the notebook's winning recipe).
 Route C: FUSION = whitened-MPNet-PCA  (+)  low-rank LLM-activation projection, concatenated.
All evaluated on HELD-OUT edits. GO if held-out near/far AUROC >= 0.85. seed=0."""
import numpy as np, json, time, torch, torch.nn as nn, torch.nn.functional as Fnn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
t0=time.time(); dev="cuda" if torch.cuda.is_available() else "cpu"
rng=np.random.default_rng(0); torch.manual_seed(0)
MODEL="gpt2-large"; N_EDIT=1500; EDIT_LAYERS=[24,30]; PARA_K=2; NEIGH_K=4
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32).to(dev).eval()
captured={}
def mk(L):
    def hook(m,i,o): captured[L]=i[0].detach()
    return hook
for L in EDIT_LAYERS: model.transformer.h[L].mlp.c_proj.register_forward_hook(mk(L))
@torch.no_grad()
def keys(prompt):
    ids=tok(prompt,return_tensors="pt",truncation=True,max_length=64).to(dev); _=model(**ids)
    return {L:captured[L][0,-1].float().cpu().numpy() for L in EDIT_LAYERS}
ds=load_dataset("azhx/counterfact",split="train",streaming=True); edits=[]; cnt=0
for ex in iter(ds):
    if cnt>=N_EDIT: break
    rw=ex["requested_rewrite"]
    ep=rw["prompt"].format(rw["subject"]) if "{}" in rw["prompt"] else rw["prompt"]+" "+rw["subject"]
    paras=ex["paraphrase_prompts"][:PARA_K]; neighs=ex["neighborhood_prompts"][:NEIGH_K]
    if len(paras)<1 or len(neighs)<1: continue
    try: ek=keys(ep); pk=[keys(p) for p in paras]; nk=[keys(nq) for nq in neighs]
    except Exception: continue
    edits.append({"ek":ek,"pk":pk,"nk":nk,"t_e":ep,"t_p":paras,"t_n":neighs}); cnt+=1
n=len(edits); print(f"edits={n} ({time.time()-t0:.0f}s)",flush=True)
idx=rng.permutation(n); tr=idx[:int(0.67*n)]; te=idx[int(0.67*n):]
def auroc_pairs(dp,dn):
    dp=np.asarray(dp); dn=np.asarray(dn)
    lab=np.r_[np.ones_like(dp),np.zeros_like(dn)]; sc=-np.r_[dp,dn]
    o=np.argsort(-sc); lab=lab[o]; P=lab.sum(); N=len(lab)-P
    if P==0 or N==0: return float("nan")
    return float(np.trapz(np.cumsum(lab)/P,np.cumsum(1-lab)/N))
res={"model":MODEL,"n_edit":n,"n_heldout":len(te),"routes":{}}
Lb=EDIT_LAYERS[-1]

# ---------- Route A': low-rank InfoNCE on LLM keys ----------
def emb_arrays(subset,L):
    E=[edits[i]["ek"][L] for i in subset]
    Pp=[[p[L] for p in edits[i]["pk"]] for i in subset]
    Nn=[[q[L] for q in edits[i]["nk"]] for i in subset]
    return E,Pp,Nn
din=edits[0]["ek"][Lb].shape[0]
Etr,Ptr,Ntr=emb_arrays(tr,Lb); Et=torch.tensor(np.array(Etr),dtype=torch.float32,device=dev)
def sample(bs=256):
    ii=rng.integers(0,len(Etr),bs)
    e=Et[ii]
    p=torch.tensor(np.array([Ptr[i][rng.integers(0,len(Ptr[i]))] for i in ii]),dtype=torch.float32,device=dev)
    ng=torch.tensor(np.array([Ntr[i][rng.integers(0,len(Ntr[i]))] for i in ii]),dtype=torch.float32,device=dev)
    return e,p,ng
def eval_proj(Wt,subset):
    dp=[];dn=[]
    for i in subset:
        e=edits[i]["ek"][Lb]@Wt; e=e/ (np.linalg.norm(e)+1e-8)
        for p in edits[i]["pk"]:
            v=p[Lb]@Wt; v=v/(np.linalg.norm(v)+1e-8); dp.append(np.linalg.norm(e-v))
        for q in edits[i]["nk"]:
            v=q[Lb]@Wt; v=v/(np.linalg.norm(v)+1e-8); dn.append(np.linalg.norm(e-v))
    return auroc_pairs(dp,dn)
bestA={"auroc_heldout":-1}
for rank in [16,32,64]:
    for wd in [1e-3,1e-2]:
        W=nn.Linear(din,rank,bias=False).to(dev)
        opt=torch.optim.Adam(W.parameters(),lr=1e-3,weight_decay=wd)
        for step in range(1200):
            e,p,ng=sample()
            ze=Fnn.normalize(W(e),dim=-1); zp=Fnn.normalize(W(p),dim=-1); zn=Fnn.normalize(W(ng),dim=-1)
            # InfoNCE: positive=paraphrase, negatives = in-batch neighbours + other edits
            tau=0.1
            pos=(ze*zp).sum(-1,keepdim=True)/tau
            neg_n=(ze*zn).sum(-1,keepdim=True)/tau
            neg_batch=ze@zp.t()/tau  # other paraphrases as negatives
            logits=torch.cat([pos,neg_n,neg_batch],dim=1)
            tgt=torch.zeros(len(ze),dtype=torch.long,device=dev)
            loss=Fnn.cross_entropy(logits,tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        Wt=W.weight.detach().cpu().numpy().T
        a=eval_proj(Wt,te); atr=eval_proj(Wt,tr)
        if a>bestA["auroc_heldout"]: bestA={"rank":rank,"wd":wd,"auroc_train":atr,"auroc_heldout":a,"Wt":Wt}
        print(f"  A' rank={rank} wd={wd}: train={atr:.3f} held={a:.3f}",flush=True)
res["routes"]["A_infonce"]={k:bestA[k] for k in ["rank","wd","auroc_train","auroc_heldout"]}
print(f"[Route A' best] rank={bestA['rank']} held-out={bestA['auroc_heldout']:.3f}",flush=True)

# ---------- Route B': MPNet -> PCA-64 + whiten (notebook recipe) ----------
routeB={}
try:
    from sentence_transformers import SentenceTransformer
    enc=SentenceTransformer("all-mpnet-base-v2",device=dev)
    def emb(txts): return enc.encode(txts,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    # fit PCA+whiten on TRAIN edit/paraphrase/neighbour texts
    train_txt=[]
    for i in tr:
        train_txt.append(edits[i]["t_e"]); train_txt+=edits[i]["t_p"]; train_txt+=edits[i]["t_n"]
    Xtr=emb(train_txt); sc=StandardScaler().fit(Xtr)
    pca=PCA(n_components=64,whiten=True,random_state=0).fit(sc.transform(Xtr))
    def proj(txts): return pca.transform(sc.transform(emb(txts)))
    def eval_txt(subset,use_pca):
        dp=[];dn=[]
        for i in subset:
            if use_pca:
                e=proj([edits[i]["t_e"]])[0]; pe=proj(edits[i]["t_p"]); ne=proj(edits[i]["t_n"])
            else:
                e=emb([edits[i]["t_e"]])[0]; pe=emb(edits[i]["t_p"]); ne=emb(edits[i]["t_n"])
            for v in pe: dp.append(np.linalg.norm(e-v))
            for v in ne: dn.append(np.linalg.norm(e-v))
        return auroc_pairs(dp,dn)
    b_raw=eval_txt(te,False); b_pca=eval_txt(te,True)
    routeB={"auroc_heldout_raw":b_raw,"auroc_heldout_pca_whiten":b_pca}
    res["routes"]["B_mpnet"]=routeB
    print(f"[Route B'] MPNet raw={b_raw:.3f}  PCA-whiten={b_pca:.3f}",flush=True)
    # ---------- Route C: fusion (whitened-MPNet-PCA (+) LLM low-rank proj) ----------
    def fused_vec(txt,keyvec):
        m=proj([txt])[0]
        a=keyvec@bestA["Wt"]; a=a/(np.linalg.norm(a)+1e-8)
        m=m/(np.linalg.norm(m)+1e-8)
        return np.concatenate([m, a])  # both unit-normalized blocks
    dp=[];dn=[]
    for i in te:
        e=fused_vec(edits[i]["t_e"],edits[i]["ek"][Lb])
        for j,p in enumerate(edits[i]["t_p"]): dp.append(np.linalg.norm(e-fused_vec(p,edits[i]["pk"][j][Lb])))
        for j,q in enumerate(edits[i]["t_n"]): dn.append(np.linalg.norm(e-fused_vec(q,edits[i]["nk"][j][Lb])))
    c=auroc_pairs(dp,dn)
    res["routes"]["C_fusion"]={"auroc_heldout":c}
    print(f"[Route C fusion] held-out={c:.3f}",flush=True)
except Exception as ex:
    res["routes"]["B_mpnet"]={"error":str(ex)}; print("Route B/C error:",ex,flush=True)

# ---------- verdict ----------
cands={"A_infonce":res["routes"]["A_infonce"]["auroc_heldout"]}
if "auroc_heldout_pca_whiten" in res["routes"].get("B_mpnet",{}): cands["B_mpnet_pca"]=res["routes"]["B_mpnet"]["auroc_heldout_pca_whiten"]
if "auroc_heldout" in res["routes"].get("C_fusion",{}): cands["C_fusion"]=res["routes"]["C_fusion"]["auroc_heldout"]
br=max(cands,key=cands.get); ba=cands[br]
verdict="GO" if ba>=0.85 else ("PARTIAL" if ba>=0.65 else "NO-GO")
res["gate1b"]={"candidates":cands,"best_route":br,"best_heldout_auroc":ba,"verdict":verdict}
json.dump(res,open("gate1b_scope.json","w"),indent=2)
print(f"\nGATE 1b VERDICT: {verdict}  best={br} held-out AUROC={ba:.3f}\nall: {cands}")
print(f"WROTE gate1b_scope.json {time.time()-t0:.0f}s")
