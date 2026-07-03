"""Independent numerical verification of Propositions 1-2 (SI Note 8).
Prop 1: participation-ratio tail bound holds (ratio <= 1) over random far configs.
Prop 2: N_eff(sigma) is monotone nondecreasing, checked at 60-digit precision.
Reproduces the zero-violation results reported in SI Note 8. seed=20260703."""
import numpy as np
from mpmath import mp, mpf, exp as mpexp
rng = np.random.default_rng(20260703)
Ksig = lambda t2, s: np.exp(-t2/(2*s**2))

# Prop 1: 20,000 configs, worst-case aligned weights
ratios, neff_over_F = [], []
for _ in range(20000):
    d = rng.uniform(0.5, 3.0); F = rng.integers(5, 200)
    r = d + rng.exponential(0.6, size=F); sig = rng.uniform(0.2, 2.0)*d
    w = Ksig(r**2, sig); Neff = (w.sum()**2)/np.sum(w**2)
    ratios.append(w.sum()/(Neff*Ksig(d**2, sig))); neff_over_F.append(Neff/F)
ratios = np.array(ratios)
print(f"Prop 1: violations={(ratios>1+1e-12).sum()} max_ratio={ratios.max():.4f} mean={ratios.mean():.4f}")

# Prop 2: 3,000 far-sets x 40 sigmas at 60-digit precision
mp.dps = 60
def Neff_mp(r, s):
    w = [mpexp(-mpf(ri)**2/(2*mpf(s)**2)) for ri in r]
    return sum(w)**2 / sum(wi*wi for wi in w)
grid = np.linspace(0.05, 4.0, 40); viol = 0; mind = mpf(1e9)
for _ in range(3000):
    d = rng.uniform(0.5, 3.0); F = rng.integers(5, 60); r = d + rng.exponential(0.6, size=F)
    vals = [Neff_mp(r, s) for s in grid]
    m = min(vals[i+1]-vals[i] for i in range(len(vals)-1))
    if m < -mpf('1e-40'): viol += 1
    mind = min(mind, m)
print(f"Prop 2: violations={viol} min_increment={mp.nstr(mind,3)}")
