# The Resolution Calibration Principle: A Certified Bandwidth for Kernel Fields

R. Negulescu (The Informational Buildup Foundation, IBF) · C. Bereanu (The Simion Stoilow Institute of Mathematics of the Romanian Academy, IMAR)

A kernel field reads a metric space through a superposition of Gaussian kernels at
weighted sources — kernel density estimators, Nadaraya–Watson regressors, radial-basis
interpolants, Gaussian-process readouts, and the correction fields of model editing are
all instances. Their behaviour is governed by one number, the bandwidth σ, almost always
chosen by labelled search. This work shows that for a kernel field the bandwidth can
instead be **calibrated** in closed form against an interference budget:

    σ*(A, ε, d) = d / sqrt(2 ln(A/ε))

the largest bandwidth whose aggregate far-source interference past a radius d is provably
below a budget ε, with the effective interfering mass A **measured** from the field rather
than tuned. The central structural result is that this measured mass is nondecreasing in σ,
so measuring it once is provably conservative and the certificate holds without any tuning.

## Contents

| File | Description |
|------|-------------|
| `rcp_paper.pdf` | Main manuscript (5 pages) |
| `rcp_si.pdf` | Supplementary Information (7 Notes: full proofs, robustness, selective-prediction, reproducibility) |
| `rcp_paper.tex`, `rcp_si.tex` | LaTeX-style source (custom markup consumed by the builders) |
| `build_pdf.py`, `build_si.py` | Offline builders — pure-Python (ReportLab + matplotlib math engine), no LaTeX toolchain required |
| `figures/` | The six figures (main: fig2, fig5, fig6, fig7; SI: fig1, fig3) |
| `experiments/` | The experiment scripts behind every reported number (certificate, δ-ball, p99/max-A, purity, growth, modality sweep, GRACE boundary) |
| `results/` | Released result files (JSON) with the exact per-experiment outputs the tables and figures are drawn from |

### Experiments

| Script | Produces |
|--------|----------|
| `p99cert.py`, `maxacert.py` | Per-query interference tail and zero-exceedance certificates on the three frozen backbones (ResNet-18/50, ViT-B/16) |
| `delta_cert.py` | Uniform-over-region δ-ball certificate radii |
| `purity.py` | Weighted near-set class purity (Table 1 near-purity column) |
| `revexp.py` | Sensitivity sweep over ε and the radius percentile |
| `modality_sweep.py` | Four non-image modalities — DBpedia, AG-News, Yahoo, Covertype (Table 2) |
| `grace_defer.py`, `grace_sweep.py`, `grace_editsite.py` | The GRACE LLM-edit-scope boundary (Section 6 honest negative) |

Result files in `results/` are named `<experiment>_<backbone|corpus>.json` and carry the
σ values, tails, accuracies, and seeds; every number in the manuscript traces to one of them.

## Rebuilding

Both PDFs regenerate offline with no LaTeX installation:

    pip install reportlab matplotlib pypdfium2 pillow
    python build_pdf.py    # -> rcp_paper.pdf
    python build_si.py     # -> rcp_si.pdf

Equations are rendered through matplotlib's Computer-Modern math engine onto the
parchment ground and composited by ReportLab; the figures are the PNGs in `figures/`.

## Reproducibility

All reported quantities — the per-query interference tail, zero-exceedance certificates,
the multi-backbone and multi-modality tables — are computed on frozen features with fixed
seeds. SI Note 7 gives the exact definitions (standardization, distance metric, subsample
sizes for d and A, the ε=0.05 budget, and the grids for every experiment family).
