# Reproducibility manifest

This manifest pins the software, data, hardware, seeds, and commands used to produce
the released numerical outputs and the manuscript figures.

## Manuscript commit
The PDFs in this repository (`rcp_main.pdf`, `rcp_si.pdf`) correspond to this
repository commit. The main text is 8 pages (≈7 pages of content; the 29-entry
bibliography spills to page 8, which does not count toward the NMI display/length limit).

## Software environment
- Python 3.12.3, CUDA 12.8
- torch 2.8.0+cu128, transformers 4.57.6, adapters 1.3.0, datasets 4.8.4
- numpy 2.1.2, pandas 3.0.2, scipy 1.17.1, pyarrow, scikit-learn, matplotlib
- See `requirements.txt`. The attention / HotpotQA / 2Wiki experiments require a
  CUDA GPU; the CIFAR, Laplace-generality, and geometry experiments run on CPU.

## Hardware
- Attention/QA experiments: single NVIDIA GeForce RTX 5090 (32 GB), CUDA 12.8.
- Wall time: unified HotpotQA sweep+audit ≈ 25 min; 3-seed 2×2 grid ≈ 55 min.

## Models, adapters, datasets (identifiers)
- Readers: `bert-base-uncased`, `roberta-base` (Hugging Face).
- Adapters: `AdapterHub/bert-base-uncased-pf-hotpotqa`,
  `AdapterHub/roberta-base-pf-hotpotqa` (loaded via the `adapters` library).
- ProtoNet: 4-layer conv, trained on Omniglot (script in `experiments/`).
- Datasets: `hotpot_qa` config `distractor` (HF); 2WikiMultiHopQA via the
  schema-matched parquet mirror `scholarly-shadows-syndicate/2WikiMultihopQA_with_q_gpt35`
  (the script-based HF loaders for 2Wiki are deprecated and no longer load).
- CIFAR-100 penultimate features from frozen ResNet-18, ResNet-50, ViT-B/16.

NOTE ON REVISION PINNING: the Hugging Face model/adapter/dataset identifiers above
are named but not pinned to individual commit hashes; to pin exactly, pass
`revision=<hash>` to `from_pretrained` / `load_dataset` for the snapshot you resolve.
The continued-trained canonical reader is not released as a checkpoint; it is
reproduced deterministically from `bert-base-uncased` + the pf-hotpotqa adapter by
`experiments/hotpot_unified.py` at seed 0 (see below).

## Seeds
- Unified HotpotQA training sweep: seeds {0,1,2} × λ ∈ {0, 0.5, 2, 8}; the full
  certificate audit uses the canonical λ=0, seed 0 reader.
- Cross-model × cross-dataset grid (`replicate_grid.py`): seeds {0,1,2} per cell.
- All example subsampling uses `np.random.RandomState(seed)`; controls bootstrap
  with 2,000 resamples.

## Key parameters (attention/QA)
- MAXLEN=512, EPS=0.05 (ε budget), TAU0=8.0 (τ₀=√d_k, d_k=64), LAYERS=[9,10,11,12],
  N_TRAIN=1500, EPOCHS=3, LR=1e-4, BATCH=4.

## Commands
```
pip install -r requirements.txt
# Additive certificate frontier (vision/text/tabular), CPU:
python experiments/laplace_gen.py
python experiments/geom_stress.py
python experiments/grow_baselines.py
# Attention / HotpotQA (GPU):
python experiments/hotpot_unified.py            # audit + 3-seed training sweep
python experiments/hotpot_unified_controls.py   # random / top-decile controls
python experiments/replicate_grid.py            # 2x2 grid, 3 seeds/cell
```

## Output files
Numerical outputs are in `results/` (one JSON per experiment); the per-head audit
tables are released as parquet checkpoints where applicable. Figure rasters are in
`figs/`. Each script writes the JSON named after it (e.g. `results/replicate_grid.json`).
