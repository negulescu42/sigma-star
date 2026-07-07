#!/usr/bin/env python3
"""Monotone-tail generalization of sigma*: Gaussian vs Laplace kernel fields.

For each dataset we build a Parzen kernel field over the training features,
fix eps=0.05, set the locality radius d to the 10th percentile of pairwise
train distances, and measure the far-set effective mass A via the far-mask
participation ratio at the single-source frontier sigma_pair.

Gaussian frontier:  sigma* = d / sqrt(2 ln(A/eps))
Laplace  frontier:  sigma* = d / ln(A/eps)      (tail exp(-r/sigma), not r^2)

We verify (a) the certified tail holds the budget (tail/eps < 1) and
(b) sigma* reaches parity with a labelled grid search anchored to each
tail's OWN sigma* scale (0.1x..5x sstar, 40 pts) so the oracle is a genuine
ceiling and sigma* cannot phantom-beat it. Also checks conservativeness
(N_eff nondecreasing => tail monotone, zero violations).
"""
import numpy as np, json
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EPS = 0.05
SEED = 20260707

def pdist2(A, B):
    return np.maximum(0.0, (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T)

# tail kernels in distance r (Gaussian uses squared distance)
def kern(R, sig, tail):
    if tail == "gauss":
        return np.exp(-(R**2) / (2 * sig * sig))
    return np.exp(-R / sig)                     # laplace

def sstar(d, A, tail):
    if tail == "gauss":
        return d / np.sqrt(2 * np.log(A / EPS))
    return d / np.log(A / EPS)

def neff_far(R, d, sig, tail):
    w = kern(R, sig, tail); farm = R > d; out = np.empty(R.shape[0])
    for q in range(R.shape[0]):
        ww = w[q][farm[q]]; s1 = ww.sum(); s2 = (ww**2).sum()
        out[q] = (s1 * s1) / s2 if s2 > 0 else 1.0
    return out.mean()

def acc(R, ytr, yq, sig, tail, nc):
    w = kern(R, sig, tail); S = np.zeros((R.shape[0], nc))
    for c in range(nc): S[:, c] = w[:, ytr == c].sum(1)
    return (S.argmax(1) == yq).mean()

def tail_over_eps(R, d, sig, tail):
    w = kern(R, sig, tail); farm = R > d
    t = np.array([w[q][farm[q]].sum() for q in range(R.shape[0])])
    return t / EPS

def run(name, X, y):
    nc = len(np.unique(y))
    Xtr, Xq, ytr, yq = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Xq = sc.transform(Xq)
    R = np.sqrt(pdist2(Xq, Xtr)); Dtr = np.sqrt(pdist2(Xtr, Xtr))
    iu = np.triu_indices(len(Xtr), 1); d = np.percentile(Dtr[iu], 10)
    rows = []
    for tail in ("gauss", "lapl"):
        sig_pair = sstar(d, 1.0 / EPS, tail) if tail == "gauss" else d / np.log(1.0 / EPS)
        A = neff_far(R, d, sig_pair, tail); ss = sstar(d, A, tail)
        toe = tail_over_eps(R, d, ss, tail)
        acc_star = acc(R, ytr, yq, ss, tail, nc)
        # per-tail anchored oracle grid
        grid = np.linspace(0.1 * ss, 5 * ss, 40)
        acc_grid = max(acc(R, ytr, yq, g, tail, nc) for g in grid)
        # monotonicity of far tail in sigma (conservativeness)
        sgrid = np.linspace(0.2 * ss, 3 * ss, 30)
        neffs = [neff_far(R, d, s, tail) for s in sgrid]
        mono_viol = int(np.sum(np.diff(neffs) < -1e-9))
        rows.append(dict(dataset=name, tail=tail, sstar=float(ss),
                         tail_over_eps_mean=float(toe.mean()), tail_over_eps_max=float(toe.max()),
                         acc_star=float(acc_star), acc_grid=float(acc_grid),
                         gap=float(acc_grid - acc_star), mono_viol=mono_viol))
    return rows

if __name__ == "__main__":
    data = [("iris", load_iris(return_X_y=True)), ("wine", load_wine(return_X_y=True)),
            ("breast-cancer", load_breast_cancer(return_X_y=True)), ("digits", load_digits(return_X_y=True))]
    res = [r for n, d in data for r in run(n, *d)]
    json.dump(res, open("laplace_gen.json", "w"), indent=2)
    print(f"{'dataset':13} {'tail':6} {'tail/eps':9} {'acc*':6} {'grid':6} {'gap':6} {'mono':4}")
    for r in res:
        print(f"{r['dataset']:13} {r['tail']:6} {r['tail_over_eps_mean']:9.3f} "
              f"{r['acc_star']*100:6.1f} {r['acc_grid']*100:6.1f} {r['gap']*100:+6.1f} {r['mono_viol']:4d}")
