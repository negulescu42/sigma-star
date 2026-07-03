
"""Generalization sweep: reproduce the certificate across MULTIPLE non-vision corpora, each with
5 seeds, to turn the single DBpedia point into a modality sweep with error bars.
Corpora: 3 text (DBpedia-14 ontology, AG-News-4 news topics, Yahoo-Answers-10 QA topics) embedded
with a frozen sentence transformer, + 1 tabular (covertype, 7 forest-cover classes, 54 raw features).
For each: naive sigma_pair, certified sigma*, CV oracle, Silverman. Report mean+/-std of
{tail/eps, acc%} and the stable=>equal rate at sigma*."""
import numpy as np, torch, json, sys, subprocess
def log(*a): print(*a,flush=True)
def ensure(p):
    try: __import__(p.split("==")[0].replace("-","_"))
    except Exception: subprocess.run([sys.executable,"-m","pip","install","-q",p],check=True)
ensure("sentence_transformers"); ensure("sklearn")
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.datasets import fetch_covtype
DEV="cuda"; EPS=0.05
enc=SentenceTransformer("all-MiniLM-L6-v2",device=DEV)

def embed_text(ds_name, text_key, label_key, ncls, per_tr=300, per_te=200, split_tr="train", split_te="test"):
    ds=load_dataset(ds_name)
    def take(split,per,rng):
        cols=ds[split].column_names
        tk=text_key if text_key in cols else [c for c in cols if c in ("content","text","question_title","best_answer")][0]
        lab=np.array(ds[split][label_key]); txt=ds[split][tk]
        idx=np.concatenate([rng.choice(np.where(lab==k)[0],per,replace=False) for k in range(ncls)])
        return [txt[i] for i in idx], lab[idx]
    def make(seed):
        rng=np.random.default_rng(seed)
        trx,try_=take(split_tr,per_tr,rng); tex,tey=take(split_te,per_te,rng)
        Xtr=enc.encode(trx,batch_size=256,convert_to_numpy=True,show_progress_bar=False,normalize_embeddings=True)
        Xte=enc.encode(tex,batch_size=256,convert_to_numpy=True,show_progress_bar=False,normalize_embeddings=True)
        return Xtr,try_,Xte,tey
    return make, ncls

def make_tabular(seed_data=0):
    cov=fetch_covtype(); X=cov.data.astype(np.float32); y=cov.target-1; ncls=7
    def make(seed):
        rng=np.random.default_rng(seed)
        tr=np.concatenate([rng.choice(np.where(y==k)[0],300,replace=False) for k in range(ncls)])
        te=np.concatenate([rng.choice(np.setdiff1d(np.where(y==k)[0],tr),200,replace=False) for k in range(ncls)])
        return X[tr],y[tr],X[te],y[te]
    return make, ncls

def evaluate(Xtr,Ytr,Xte,Yte,ncls,rng):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    Xk=torch.tensor(Xtr,device=DEV,dtype=torch.float32); Yk=torch.tensor(Ytr,device=DEV)
    Xt=torch.tensor(Xte,device=DEV,dtype=torch.float32); Yt=torch.tensor(Yte,device=DEV)
    oh=torch.zeros(len(Yk),ncls,device=DEV); oh[torch.arange(len(Yk)),Yk]=1.0
    sub=Xk[rng.choice(len(Xk),min(4000,len(Xk)),replace=False)]; pdm=torch.cdist(sub,sub)
    d=torch.quantile(pdm[pdm>0],0.10).item()
    Ksig=lambda s,r: torch.exp(-r/(2*s*s))
    @torch.no_grad()
    def measure_A(sig,nq=3000):
        q=Xt[rng.choice(len(Xt),min(nq,len(Xt)),replace=False)]; v=[]
        for i in range(0,len(q),500):
            D2=torch.cdist(q[i:i+500],Xk).pow(2); fm=D2>d*d
            w=Ksig(sig,D2)*fm; s=w.sum(1); s2=(w*w).sum(1); v.append(((s*s)/(s2+1e-30)).cpu().numpy())
        return float(np.concatenate(v).mean())
    @torch.no_grad()
    def tail(sig,nq=3000):
        q=Xt[rng.choice(len(Xt),min(nq,len(Xt)),replace=False)]; t=[]
        for i in range(0,len(q),500):
            D2=torch.cdist(q[i:i+500],Xk).pow(2); fm=D2>d*d; t.append((Ksig(sig,D2)*fm).sum(1).cpu().numpy())
        return float(np.concatenate(t).mean())
    @torch.no_grad()
    def acc(sig,bs=2000):
        inv=1/(2*sig*sig); cc=0
        for i in range(0,len(Xt),bs):
            d2=torch.cdist(Xt[i:i+bs],Xk).pow(2); cc+=((torch.exp(-d2*inv)@oh).argmax(1)==Yt[i:i+bs]).sum().item()
        return cc/len(Xt)
    def cv_sigma():
        g=np.geomspace(0.3,40,30); return g[int(np.argmax([acc(s) for s in g]))]
    @torch.no_grad()
    def perfcert(sig,nq=6000):
        q=Xt[rng.choice(len(Xt),min(nq,len(Xt)),replace=False)]; inv=1/(2*sig*sig); st_n=eq=tot=0
        for i in range(0,len(q),500):
            D2=torch.cdist(q[i:i+500],Xk).pow(2); near=D2<=d*d; Kv=torch.exp(-D2*inv)
            scn=(Kv*near)@oh; sca=Kv@oh
            t2=torch.topk(scn,2,dim=1).values; st=(t2[:,0]-t2[:,1])>2*EPS
            eqp=(scn.argmax(1)==sca.argmax(1)); st_n+=st.sum().item(); eq+=(st&eqp).sum().item(); tot+=len(D2)
        return st_n/tot,(eq/st_n if st_n else float('nan'))
    sig_pair=d/np.sqrt(2*np.log(1/EPS)); A=measure_A(sig_pair); sig_star=d/np.sqrt(2*np.log(A/EPS))
    sig_cv=cv_sigma(); ndim=Xtr.shape[1]
    silver=float(np.mean(np.std(Xtr,0))*(len(Xk)*(ndim+2)/4)**(-1/(ndim+4)))
    # reviewer-requested label-free baselines: median heuristic and k-NN(7) distance
    sig_med=float(torch.median(pdm[pdm>0]))
    @torch.no_grad()
    def knn_sigma(k=7):
        tot=0.0; nb=0
        for i in range(0,len(Xk),1000):
            D=torch.cdist(Xk[i:i+1000],Xk); D,_=torch.sort(D,1)
            kk=min(k,D.size(1)-1); tot+=D[:,kk].sum().item(); nb+=D.size(0)
        return tot/nb
    sig_knn=knn_sigma(7)
    fs,se=perfcert(sig_star)
    return dict(d=d,dim=int(ndim),
        naive=[tail(sig_pair)/EPS,acc(sig_pair)*100],
        star=[tail(sig_star)/EPS,acc(sig_star)*100,fs,se],
        cv=[tail(sig_cv)/EPS,acc(sig_cv)*100],
        silver=[tail(silver)/EPS,acc(silver)*100],
        median=[tail(sig_med)/EPS,acc(sig_med)*100],
        knn7=[tail(sig_knn)/EPS,acc(sig_knn)*100])

CORPORA=[
    ("DBpedia-14 (text/ontology)", *embed_text("fancyzhx/dbpedia_14","content","label",14)),
    ("AG-News-4 (text/news)",      *embed_text("fancyzhx/ag_news","text","label",4)),
    ("Yahoo-10 (text/QA)",         *embed_text("yahoo_answers_topics","question_title","topic",10)),
    ("Covertype-7 (tabular)",      *make_tabular()),
]
SEEDS=[0,1,2,3,4]; results={}
for name,make,ncls in CORPORA:
    per_seed={k:[] for k in ["naive","star","cv","silver","median","knn7"]}; dim=None
    for s in SEEDS:
        try:
            Xtr,Ytr,Xte,Yte=make(s); rng=np.random.default_rng(1000+s)
            r=evaluate(Xtr,Ytr,Xte,Yte,ncls,rng); dim=r["dim"]
            for k in per_seed: per_seed[k].append(r[k])
        except Exception as e:
            log(f"  {name} seed {s} FAILED: {e}")
    agg={}
    for k,v in per_seed.items():
        a=np.array([x[:2] for x in v]); agg[k]=dict(mean=a.mean(0).tolist(),std=a.std(0).tolist())
    stars=np.array([x[2:] for x in per_seed["star"]]); agg["star"]["frac_stable"]=float(stars[:,0].mean()); agg["star"]["stable_eq"]=float(np.nanmean(stars[:,1]))
    agg["dim"]=dim; agg["n_classes"]=ncls; results[name]=agg
    m=agg["star"]["mean"]; log(f"{name}: dim={dim} star tail/eps={m[0]:.2f} acc={m[1]:.1f} stable=>eq={agg['star']['stable_eq']:.3f}")
json.dump(results,open("modality_sweep.json","w"),indent=2); log("DONE")
