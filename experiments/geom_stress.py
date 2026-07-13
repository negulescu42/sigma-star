
import numpy as np, json
from sklearn.datasets import make_moons
rng=np.random.default_rng(0)
EPS=0.05

def sstar_from(dists_far, d, Vmax=1.0):
    # A = Vmax * Neff over far set;  Neff=(sum w)^2/sum w^2 at reference sigma_pair
    # use reference sigma = sigma_pair = d/sqrt(2 ln(Vmax/eps))? define A at reference broad scale
    return None

def neff(w):
    s1=w.sum(); s2=(w*w).sum()
    return (s1*s1)/s2 if s2>0 else 0.0

def certify(rho, d, eps=EPS, Vmax=1.0):
    """rho: (Nq, Nsrc) fixed dissimilarities. Returns sigma*, A, tail/eps at sstar, coverage, violation."""
    far = rho>d
    # reference sigma_pair = d/sqrt(2 ln(1/eps)) (broad reference), measure A there
    sig_pair = d/np.sqrt(2*np.log(Vmax/eps))
    # A = Vmax * mean Neff of far weights at sig_pair (conservative envelope): use max over queries for one-shot
    Wfar = np.exp(-(rho**2)/(2*sig_pair**2))*far
    A = Vmax*np.median([neff(Wfar[q][far[q]]) for q in range(rho.shape[0]) if far[q].any()])
    sstar = d/np.sqrt(2*np.log(A/eps)) if A>eps else d
    # tail at sstar per query = sum far weights (Vmax units)
    Ws = np.exp(-(rho**2)/(2*sstar**2))
    tail = (Ws*far).sum(axis=1)
    tail_over_eps = np.median(tail)/eps
    viol = float(np.mean(tail>eps))
    return dict(sig_pair=float(sig_pair), A=float(A), sstar=float(sstar),
                tail_over_eps=float(tail_over_eps), viol_rate=viol,
                mean_tail=float(np.mean(tail)))

def decisions(rho, labels, sig):
    """kernel-vote argmax over classes using weights exp(-rho^2/2sig^2)."""
    W=np.exp(-(rho**2)/(2*sig**2))
    C=labels.max()+1
    scores=np.zeros((rho.shape[0],C))
    for c in range(C):
        scores[:,c]=(W*(labels==c)[None,:]).sum(axis=1)
    return scores.argmax(1)

out={}

# ---------- (1) anisotropic Gaussian family, increasing axis ratio ----------
aniso=[]
for ratio in [1.0,2.0,4.0,8.0]:
    n=400
    X=rng.standard_normal((n,2)); X[:,1]*=1.0  # base
    y=(X[:,0]>0).astype(int)  # class by x0 (the informative axis)
    # inflate the uninformative axis by ratio -> Euclidean distance dominated by noise axis
    Xa=X.copy(); Xa[:,1]*=ratio
    # queries = held-out
    Xq=rng.standard_normal((200,2)); yq=(Xq[:,0]>0).astype(int); Xqa=Xq.copy(); Xqa[:,1]*=ratio
    # Euclidean geometry
    rhoE=np.sqrt(((Xqa[:,None,:]-Xa[None,:,:])**2).sum(-1))
    dE=np.percentile(rhoE,10)
    cE=certify(rhoE,dE)
    predE=decisions(rhoE,y,cE['sstar']); accE=float((predE==yq).mean())
    # frozen Mahalanobis M = diag(1, 1/ratio^2): undo the inflation (known frozen metric)
    M=np.diag([1.0,1.0/ratio**2])
    diff=Xqa[:,None,:]-Xa[None,:,:]
    rhoM=np.sqrt(np.einsum('qsi,ij,qsj->qs',diff,M,diff))
    dM=np.percentile(rhoM,10)
    cM=certify(rhoM,dM)
    predM=decisions(rhoM,y,cM['sstar']); accM=float((predM==yq).mean())
    aniso.append(dict(ratio=ratio,
        eucl=dict(**cE,acc=accE), mahal=dict(**cM,acc=accM)))
out['anisotropic']=aniso

# ---------- (2) two-moons: Euclidean proximity != intrinsic connectivity ----------
Xm,ym=make_moons(n_samples=600,noise=0.08,random_state=1)
Xtr,ytr=Xm[:400],ym[:400]; Xq,yq=Xm[400:],ym[400:]
rhoE=np.sqrt(((Xq[:,None,:]-Xtr[None,:,:])**2).sum(-1))
dE=np.percentile(rhoE,10); cE=certify(rhoE,dE)
predE=decisions(rhoE,ytr,cE['sstar']); accE=float((predE==yq).mean())
# graph-geodesic (frozen): build kNN graph on TRAIN, out-of-sample query distance = min over
# (euclid to a train anchor) + geodesic(anchor, target); Dijkstra via sklearn
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
G=kneighbors_graph(Xtr,n_neighbors=8,mode='distance')
Gsym=G.maximum(G.T)
geo=dijkstra(Gsym,directed=False)  # (Ntr,Ntr)
# query->train: connect query to its nearest train anchor(s), add euclid
knn=8
qd=np.sqrt(((Xq[:,None,:]-Xtr[None,:,:])**2).sum(-1))
anch=np.argsort(qd,axis=1)[:,:knn]
rhoG=np.full((Xq.shape[0],Xtr.shape[0]),np.inf)
for q in range(Xq.shape[0]):
    for a in anch[q]:
        cand=qd[q,a]+geo[a]
        rhoG[q]=np.minimum(rhoG[q],cand)
finite=np.isfinite(rhoG)
rhoG[~finite]=rhoG[finite].max()*1.5
dG=np.percentile(rhoG,10); cG=certify(rhoG,dG)
predG=decisions(rhoG,ytr,cG['sstar']); accG=float((predG==yq).mean())
# agreement of geodesic-RCP with euclid-RCP decisions
agree=float((predE==predG).mean())
out['two_moons']=dict(eucl=dict(**cE,acc=accE),
                      geodesic=dict(**cG,acc=accG),
                      eucl_geo_decision_agreement=agree)

# ---------- (3) densification behaviour: isotropic vs frozen-Mahalanobis ----------
dens=[]
for n in [200,800,3200]:
    ratio=4.0
    X=rng.standard_normal((n,2)); y=(X[:,0]>0).astype(int); Xa=X.copy(); Xa[:,1]*=ratio
    Xq=rng.standard_normal((200,2)); yq=(Xq[:,0]>0).astype(int); Xqa=Xq.copy(); Xqa[:,1]*=ratio
    rhoE=np.sqrt(((Xqa[:,None,:]-Xa[None,:,:])**2).sum(-1)); dE=np.percentile(rhoE,10); cE=certify(rhoE,dE)
    M=np.diag([1.0,1.0/ratio**2]); diff=Xqa[:,None,:]-Xa[None,:,:]
    rhoM=np.sqrt(np.einsum('qsi,ij,qsj->qs',diff,M,diff)); dM=np.percentile(rhoM,10); cM=certify(rhoM,dM)
    dens.append(dict(n=n, eucl_sstar=cE['sstar'], eucl_tail_over_eps=cE['tail_over_eps'], eucl_viol=cE['viol_rate'],
                     mahal_sstar=cM['sstar'], mahal_tail_over_eps=cM['tail_over_eps'], mahal_viol=cM['viol_rate']))
out['densification']=dens

json.dump(out, open('geom_stress.json','w'), indent=1)
print(json.dumps(out, indent=1)[:2500])
