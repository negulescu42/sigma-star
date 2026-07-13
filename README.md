# The Resolution Calibration Principle: Certified Locality for Kernel and Gibbs Fields

R. Negulescu (The Informational Buildup Foundation, IBF) · M. Tănase (Bucharest, Romania) · C. Bereanu (The Simion Stoilow Institute of Mathematics of the Romanian Academy, IMAR)

A kernel field reads a metric space through a superposition of distance-decaying
kernels at weighted sources — kernel density estimators, Nadaraya–Watson regressors,
radial-basis interpolants, Gaussian-process readouts, prototype classifiers, and softmax
attention are all instances. Their behaviour is governed by one resolution parameter (the
bandwidth σ, or a temperature τ), almost always chosen by labelled search. This work shows
that the resolution can instead be **calibrated** in closed form against an interference
budget:

    σ*(A, ε, d) = d / sqrt(2 ln(A/ε))

the largest bandwidth whose aggregate far-source interference past a radius d is provably
below a budget ε, with the effective interfering mass A **measured** from the field rather
than tuned. The measured mass is nondecreasing in σ, so one measurement is provably
conservative and the certificate holds without tuning. A Gibbs form extends the principle
to normalised readouts (prototype networks, attention); a Rényi family tightens the mass
bound; a split-conformal wrapper gives finite-sample coverage for future queries.

## Contents

| File | Description |
|------|-------------|
| `rcp_main.pdf` | Main manuscript (6 pages) |
| `rcp_si.pdf` | Supplementary Information (16 Notes: full proofs, Gibbs/Rényi/conformal extensions, robustness, reproducibility) |
| `rcp_main.tex`, `rcp_si.tex` | LaTeX source (compile with `pdflatex`, two passes each) |
| `figs/` | The manuscript figures, referenced by the sources as `figs/fig_<id>.png` |
| `experiments/` | The experiment scripts behind every reported number |
| `results/` | Released result files (JSON) with the exact per-experiment outputs the tables and figures draw from |

## Rebuilding

    pdflatex rcp_main.tex && pdflatex rcp_main.tex
    pdflatex rcp_si.tex   && pdflatex rcp_si.tex

## Key experiments

| Script | Produces |
|--------|----------|
| `experiments/l1_cert.py` | Vector ℓ₁ decision certificate and coverage-matched selective-prediction accuracies on the three frozen backbones |
| `experiments/grow_baselines.py` | Label-free median / k-NN bandwidth scales recomputed at each densification stage (SI Note 15 table) |
| `experiments/geom_stress.py` | Frozen-geometry stress test — anisotropic / two-moons / densification, Euclidean vs Mahalanobis / geodesic (SI Note 14 figure) |
| `experiments/renyi_conformal.py` | Rényi D_q mass family and split-conformal query coverage |
| `experiments/gibbs_attention.py`, `experiments/gibbs_proto.py` | Gibbs far-mass certificate on BERT/GPT-2 attention and a ProtoNet |
| `experiments/modality_sweep.py`, `experiments/grow_backbone.py` | Multi-modality and growing-memory re-calibration tables |

Result files in `results/` carry the σ/τ values, tails, accuracies, and seeds; every number
in the manuscript traces to one of them.

## Reproducibility

All reported quantities are computed on frozen features with fixed seeds. The Methods
section and SI give the exact definitions (standardization, distance metric, subsample
sizes for d and A, the ε=0.05 budget, and the grids for every experiment family).
