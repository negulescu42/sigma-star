# Response to Referee — *The Resolution Calibration Principle: A Certified Bandwidth for Kernel Fields*

We thank the referee for an unusually careful report. We are particularly grateful that they independently re-derived and numerically stress-tested Propositions 1 and 2 before writing — 20,000 configurations for the tail bound, 3,000 trials at 60-digit precision for the monotonicity, both with zero violations. This is the strongest form of scrutiny a theoretical result can receive, and we have made it a permanent part of the paper (new **SI Note 8**, with a reproduction script in the repository). We read the report's verdict as: the mathematics is sound, novel, and honest, and the reservations are about **positioning**. We have acted on the positioning points and, where we respectfully disagree, we say why. Two of the referee's concerns rested on material they did not receive (the Supplementary Information and a live repository link); both are now resolved.

Throughout, "§N" refers to the numbered concerns in the report.

---

## Points we accept and have implemented

### §3.3 — "Corollary 1 floors faithfulness, not accuracy."
**Accepted; made unmissable.** The referee is right that the performance floor guarantees *decision equality with the near-field vote*, not correctness against ground truth, and that near-purity is a *plurality* (0.25–0.47), not a majority. The manuscript already said "the certificate guarantees *which* sources decide, not that they decide correctly," but the referee is correct that this deserved to be foregrounded rather than buried. We have kept and sharpened this framing at both the Corollary and in the flagship Results, and the abstract's accuracy claim remains stated as an *empirical* match to grid search ("within 1–5 points"), never as a theorem. The certificate is a faithfulness guarantee; accuracy is a separately measured outcome. We believe the paper is now unambiguous on this.

### §3.4 — "Elevate Fig. 4 ('the budget travels') toward the front."
**Accepted.** This is, as the referee notes, the one place where the certificate does something a labelled search structurally *cannot* — re-certify for free as the field grows, where a frozen grid-search bandwidth is silently wrong for the field it is deployed on. The abstract and introduction now lead with re-certification under growth as the central practical payoff, and §"The Budget Travels" carries the demonstration (7-point margin against a deliberately frozen comparator, honestly stated as tracking a per-stage re-tuned oracle to within a fraction of a point, not dominating it).

### §3.5 — "Run the missing baselines (Sheather–Jones, GP marginal-likelihood)."
**Accepted, and run.** We were able to compute both on the three frozen backbones with the identical evaluation used for the tables. The result strengthens the paper: the two objective-mismatched selectors miss the interference budget in **opposite** directions.

| Selector | Objective | σ vs σ\* | tail/ε | accuracy |
|---|---|---|---|---|
| **Sheather–Jones** | ISE plug-in | ≈ 0.02–0.04× (near-delta) | ≈ 0 | **chance (1%)** |
| **GP marginal-likelihood** | one-hot LML | ≈ 11–15× | ≈ 3×10⁵ | 24–62% |
| **σ\*** (ours) | interference budget | 1× | 0.08–0.16 | 52–72% |

Sheather–Jones optimizes per-dimension integrated squared error, which in high-dimensional standardized feature space collapses to a near-delta kernel and destroys the classifier; GP marginal likelihood prefers wide kernels and blows the budget by five to six orders of magnitude. Neither *sees* the interference budget, which is precisely the paper's thesis. This is now a **measured** sentence in Related Work, not a cited claim. (Script: `experiments/extra_baselines_a2.py`; data: `results/extra_baselines_a2.json`.)

### SI + repository (§3.5, "could not certify the load-bearing SI claims")
**Resolved.** The referee reviewed a package without the SI or a working code link, and correctly declined to certify claims they could not check. Both are now in hand: the full Supplementary Information (eight Notes, including the referee's own verification as Note 8) accompanies the manuscript, and the repository is live and populated at **github.com/negulescu42/sigma-star** — 49 files, including every experiment script, released result JSONs, and the two builders, so every number in the paper is reproducible end-to-end.

### The verification gift (Rényi-2 framing of Proposition 2)
**Accepted with thanks.** The referee's observation that the far-weight family ω_i = exp(−r_i²/2σ²) is a tempered family (β = 1/2σ²) and that N_eff is exactly its inverse-Simpson / **Rényi-2 diversity** gives Proposition 2 a one-line reading: monotonicity is a diversity index rising as a distribution is tempered toward uniform. We have recorded this in SI Note 8 ("A shorter route to Proposition 2"), crediting it as a connection to the diversity-index literature.

---

## Points we respectfully contest

### §3.1 — "The model-editing motivation is a bait-and-switch."
**We disagree, though we have sharpened the framing.** A bait-and-switch hides where a method fails. This paper does the opposite: the **abstract itself** names the GRACE boundary ("while, on a GRACE-style LLM editor, it maps exactly where the principle's own assumption runs out"), and an entire section (§6, "A Boundary: LLM Edit Scope") exists for no other purpose than to report, quantitatively (AUROC 0.60, at or below chance in deep layers), the regime where the certificate's near/far assumption (A2) does not hold in raw LLM key space. We have moved this signposting even earlier, but we do not accept the characterization: a paper that leads with its own limitation is not baiting. What the referee may be reacting to is the *order* of the older draft, which motivated with editing before delivering on vision; the reframing (§3.4) fixes that by leading with the re-certification payoff, which the vision experiments fully support.

### §3.2 — "The KDE comparison grades density selectors on the wrong objective — a strawman."
**Partly fair, and we now say so explicitly — but it is not a strawman.** The referee is correct that Silverman/Scott/Sheather–Jones optimize integrated squared error against a *density*, a different objective from interference, and the revised Related Work states this directly. But two facts keep the comparison legitimate rather than a strawman: (i) practitioners *do* reach for Silverman's rule as a default bandwidth for exactly these kernel constructions — it is the scikit-learn / seaborn default — so showing that this default overshoots the interference budget by 10⁵× is a real-world warning, not a rigged fight; and (ii) we do not claim the density selectors are *wrong at their own job* — we claim they are *blind to the budget*, which the new opposite-direction result (§3.5) demonstrates cleanly. We have softened the rhetoric ("five to six orders" is now scoped to the vision fields, where it holds) without removing the comparison.

### §3.6 — "The certificate is vacuous without a label-free A2 diagnostic."
**We disagree on 'vacuous,' we tried the constructive suggestion, and we report the result honestly — it did not work.** First, on 'vacuous': choosing a locality radius *d* is standard and unavoidable in certified robustness (it is the perturbation-ball radius); a per-query certificate conditioned on a stated geometry is exactly what interval-bound and randomized-smoothing certificates provide, and no one calls those vacuous. The certificate holds whenever A2 holds, and A2 is an explicit, checkable modeling assumption.

Second, and more usefully, we took the referee's *constructive* suggestion seriously and built a candidate **label-free A2 diagnostic** — the kernel-weight fraction falling in a "twilight annulus" around the radius *d*, a pure-geometry measure of how ambiguous the near/far partition is. We tested whether it predicts the one quantity a deployer would care about: the accuracy gap between σ\* and a labelled grid search, across all eight fields (three CIFAR backbones + five public datasets).

**It does not.** Pearson correlation between the diagnostic and the σ\*-vs-grid gap is **−0.07**. Covertype has the *sharpest* boundary yet the *largest* gap (+6.3 pts); ResNet-18 has the *fuzziest* boundary yet a negligible gap (+0.5 pts). We could have reported a cherry-picked positive version; instead we state the honest finding: **the accuracy gap is small everywhere (≤ 6 points), so there is little signal for any diagnostic to predict, and this particular statistic predicts none of it.** We are therefore *not* adding a non-predictive diagnostic dressed up as a working tool — that would be exactly the kind of overclaim the rest of the paper avoids. We flag the open problem (a *learned* scope geometry, as §6 already proposes for the LLM case) as future work, which is where it honestly belongs. (Script and null result: `experiments/extra_baselines_a2.py`, `results/a2_correlation.json`.)

---

## Summary of changes

1. New **SI Note 8** — independent numerical verification of Propositions 1–2 (tail-bound tightness, N_eff monotonicity at 60-digit precision), with the referee's Rényi-2 framing and a reproduction script.
2. **Measured Sheather–Jones and GP marginal-likelihood** baselines added to Related Work — the opposite-direction budget failures.
3. **Reframing** to lead with re-certification-under-growth ("the budget travels").
4. **Faithfulness-not-accuracy** caveat foregrounded at Corollary 1 and in Results.
5. **SI and live repository** (github.com/negulescu42/sigma-star, 49 files) now accompany the manuscript.
6. Tested the proposed **A2 diagnostic**; report the honest null (r = −0.07) rather than an overclaim.

We believe the paper is stronger for this review, and that the remaining disagreements are ones of framing on which reasonable reviewers can differ — not defects in the result the referee themselves verified.
