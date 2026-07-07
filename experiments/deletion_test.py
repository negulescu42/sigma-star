#!/usr/bin/env python3
"""Far-set locality experiment (Corollary 2).

At sigma <= sigma*, deleting the entire far set must move the output by <= eps,
and no admissible far-set corruption may flip a decision with near margin > 4eps.
We test this on four public datasets at sigma* and at an over-wide 3*sigma*:

  del_shift        = || F(y) - S(y) ||_inf  (output shift when far set is deleted)
  flip_del_cert    = frac. of certified-stable (near margin > 2eps) decisions that
                     flip when the far set is deleted
  flip_cor_cert    = same, under worst-case adversarial far-set corruption
                     (all far kernel mass reassigned to the near runner-up class)

Gaussian tail, eps=0.05, d = 10th pctile pairwise train distance,
A = far-mask participation ratio at sigma_pair, sigma* = d/sqrt(2 ln(A/eps)).
"""
import numpy as np, json
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EPS = 0.05

def pdist2(A, B):
    return np.maximum(0.0, (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T)
gauss = lambda R, s: np.exp(-(R**2) / (2 * s * s))

def neff_far(R, d, sig):
    w = gauss(R, sig); farm = R > d; out = np.empty(R.shape[0])
    for q in range(R.shape[0]):
        ww = w[q][farm[q]]; s1 = ww.sum(); s2 = (ww**2).sum()
        out[q] = (s1 * s1) / s2 if s2 > 0 else 1.0
    return out.mean()

def scores(R, ytr, sig, nc, farm=None, mode="full"):
    w = gauss(R, sig).copy()
    if mode in ("near", "corrupt"):
        w_near = w.copy(); w_near[farm] = 0.0
        wf = w.copy(); wf[~farm] = 0.0
    if mode == "full":
        S = np.zeros((R.shape[0], nc))
        for c in range(nc): S[:, c] = w[:, ytr == c].sum(1)
        return S
    if mode == "near":
        S = np.zeros((R.shape[0], nc))
        for c in range(nc): S[:, c] = w_near[:, ytr == c].sum(1)
        return S
    # corrupt: near scores + all far mass pushed to near runner-up class
    Snear = np.zeros((R.shape[0], nc))
    for c in range(nc): Snear[:, c] = w_near[:, ytr == c].sum(1)
    far_mass = wf.sum(1); S = Snear.copy()
    for q in range(R.shape[0]):
        order = np.argsort(Snear[q])[::-1]
        runner = order[1] if nc > 1 else order[0]
        S[q, runner] += far_mass[q]
    return S

def run(name, X, y):
    nc = len(np.unique(y))
    Xtr, Xq, ytr, yq = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Xq = sc.transform(Xq)
    R = np.sqrt(pdist2(Xq, Xtr)); Dtr = np.sqrt(pdist2(Xtr, Xtr))
    iu = np.triu_indices(len(Xtr), 1); d = np.percentile(Dtr[iu], 10)
    sig_pair = d / np.sqrt(2 * np.log(1 / EPS)); A = neff_far(R, d, sig_pair)
    ss = d / np.sqrt(2 * np.log(A / EPS))
    out = {"dataset": name, "sstar": float(ss), "d": float(d)}
    for tag, sig in [("sstar", ss), ("wide3x", 3 * ss)]:
        farm = R > d
        F = scores(R, ytr, sig, nc, farm, "full")
        Sn = scores(R, ytr, sig, nc, farm, "near")
        Sc = scores(R, ytr, sig, nc, farm, "corrupt")
        del_shift = np.abs(F - Sn).max(1)
        Sn_s = np.sort(Sn, 1)[:, ::-1]; near_margin = Sn_s[:, 0] - Sn_s[:, 1]
        cert = near_margin > 2 * EPS
        flip_del = F.argmax(1) != Sn.argmax(1)
        flip_cor = F.argmax(1) != Sc.argmax(1)
        out[tag] = dict(del_shift_max=float(del_shift.max()),
                        del_over_eps_max=float(del_shift.max() / EPS),
                        cert_frac=float(cert.mean()),
                        flip_del_cert=float(flip_del[cert].mean()) if cert.any() else 0.0,
                        flip_cor_cert=float(flip_cor[cert].mean()) if cert.any() else 0.0,
                        n_cert=int(cert.sum()))
    return out

if __name__ == "__main__":
    data = [("iris", load_iris(return_X_y=True)), ("wine", load_wine(return_X_y=True)),
            ("breast-cancer", load_breast_cancer(return_X_y=True)), ("digits", load_digits(return_X_y=True))]
    res = [run(n, *d) for n, d in data]
    json.dump(res, open("deletion_test.json", "w"), indent=2)
    print(f"{'dataset':13} {'sig':7} {'del/eps':9} {'cert%':6} {'flip_del%':10} {'flip_cor%':10}")
    for r in res:
        for tag in ("sstar", "wide3x"):
            g = r[tag]
            print(f"{r['dataset']:13} {tag:7} {g['del_over_eps_max']:9.3f} {g['cert_frac']*100:6.1f} "
                  f"{g['flip_del_cert']*100:10.2f} {g['flip_cor_cert']*100:10.2f}")
