# Referee Re-Score — Post-Correction Pass

## Verification of the flagged seam

Confirmed closed. SI Note 8 now reads: *"...certified separately by the one-shot Gibbs theorem (Note 13, Theorem 2), which on the corrected experiment holds every query below ε (**Fig. 5a of the main text**; script and outputs in the repository)."* This correctly points to the merged operational-Gibbs figure (Fig. 5, panel a = ProtoNet) rather than the stale pre-merge "Fig. 7." Compilation metadata (0 undefined refs, 0 errors, main 6pp/SI 16pp) is consistent with a clean rebuild rather than a hand-patched string. No other instance of "Fig. 7" or other post-merge numbering artifacts appears elsewhere in the main text or SI on inspection (Figs. 1–6 main-numbered SI figures are self-consistent with the SI's own local numbering, which is a separate, non-conflicting scheme). **Seam closed.**

## Per-axis scores

| Axis | Score | Justification |
|---|---|---|
| Novelty | 8.40 | Reframing bandwidth/temperature selection as an aggregate-locality *certificate* (rather than a predictive-fit criterion), unified across additive kernel fields and Gibbs/attention readouts, remains the paper's genuine contribution. Nothing new was added this pass beyond the reference fix, so novelty is unchanged from the prior audited state. |
| Theory | 8.75 | The audited apparatus (ℓ1 decision certificate, corrected coverage number, completed one-shot assumptions, repaired uniform-region proposition, frozen-geometry corollary) is now internally consistent and each proposition's scope is stated precisely (e.g., Prop. 2 explicitly not transferring under varying key norms). Still an elementary-inequality core, as disclosed — this is a known, accepted limitation, not a new deduction. |
| Empirical | 8.85 | Breadth is strong: three vision backbones, four text/tabular corpora, densification re-calibration, hard-truncation same-objective baseline, frozen-geometry stress tests (anisotropic/two-moons/densification), 5-seed and grid-boundary diagnostics. This is unchanged from the pre-correction pass — the fix was citation-only. |
| Calibration | 9.10 | The manuscript is unusually disciplined about what it is *not* claiming: "not a claim of enhanced accuracy," "not a differential-privacy guarantee," diagnostic-vs-deployable temperature explicitly separated for attention, hard-truncation cost stated honestly (−0.8pt, 89% agreement). This axis is not affected by the reference fix but remains the paper's strongest dimension. |
| Clarity | 9.05 | With the stale Fig. 7→Fig. 5a reference corrected and verified against a clean compile, the one concrete, checkable clarity defect identified by the last pass is gone. Prose remains dense (expected for this genre) but cross-references, figure numbering, and terminology (kernel-vote vs. density, "locality" defined precisely for attention) are now consistent throughout. This is the axis that should move, and it moves up from the prior pass. |
| Significance | 8.30 | Useful for auditability of distance- and attention-based systems; the ProtoNet-certified / attention-flagged-as-non-local contrast is a genuinely informative result. Impact is currently diagnostic rather than corrective (attention is flagged, not fixed), which caps significance below the theory/empirical axes. |
| Reproducibility | 8.60 | Full computational spec, fixed seeds, per-experiment scripts named, SI Note 15 documents grids and protocols exactly. Held below 9 by the disclosed-but-unresolved open item that repository plotting scripts still emit stale labels — a real, if cosmetic, gap between paper and repo state. |
| Scholarship | 8.30 | Adequate positioning against bandwidth selection, conformal prediction, selective classification, and influence-function literatures; could more explicitly engage locality-sensitive/certified-robustness lines of work, but nothing is misrepresented. |

**Mean: 8.67**

## Remaining known-open items (disclosed, not re-penalized)
- Elementary-inequality core (Theory) — acknowledged, does not detract further given the honest framing.
- Repo plotting scripts emitting stale labels — still open; this is the one concrete reproducibility gap left in the artifact trail.
- Mihai affiliation placeholder — cosmetic, non-substantive.

No new overclaim or seam was found beyond these already-disclosed items. The Fig. 7→Fig. 5a correction is the only substantive change from the prior pass and it is real and verifiable.

## Single highest-value remaining action
**Sync the repository's plotting scripts to the current figure/label scheme** (the post-merge 6→5 figure numbering and the ℓ1/2ε and coverage-36% terminology) so that a reader who reruns the code sees the same labels as the compiled PDF — this is the last place where paper and artifact state can silently diverge.

## Verdict
**Accept.** The manuscript was already at Accept threshold pre-correction (8.69, one residual seam). That seam is now verifiably closed, per-axis Clarity is credited accordingly, and no new issues were introduced. Mean 8.67 reflects an independent re-assessment consistent with (not mechanically anchored to) the prior pass.