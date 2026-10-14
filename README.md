# Adaptive Fake News Detection on Social Media: A Gated Semantic-Topological Network with Continual Learning

This repository contains the official PyTorch implementation of the paper **"Adaptive Fake News Detection on Social Media: A Gated Semantic-Topological Network with Continual Learning"**. 

The codebase provides a modular, production-ready framework to evaluate Fake News Detection models against two compounding challenges: **Concept Drift** (evolving vocabulary) and **Algorithmic Drift** (structural collapse of propagation cascades). It features a Hybrid Gated BiGCN architecture and integrates Elastic Weight Consolidation (EWC) to mitigate catastrophic forgetting during cross-domain adaptation.

---

## Repository Structure

The codebase is strictly modularized, separating data preprocessing, model definitions, and multi-seed execution scripts.

```text
├── preprocessing/
│   ├── 01_etl_pipeline.py                 # Raw data cleaning and normalization
│   ├── 02_feature_extraction_semantic.py  # BERTweet FP16 embeddings generation
│   └── 03_feature_extraction_topological.py # PyTorch Geometric graph construction
├── models/
│   ├── semantic.py      # Semantic Baseline (BERTweet [CLS] MLP)
│   ├── topological.py   # Topological Baseline (Bi-directional GCN)
│   ├── hybrid.py        # Hybrid Gated BiGCN Architecture
│   ├── ewc.py           # Elastic Weight Consolidation (Fisher Matrix, Penalty & Per-Group Stats)
│   └── ablation.py      # Ablation Model (Hybrid Gated MLP without message passing)
├── utils.py             # Shared utilities (Seed setting, Metrics, Aggregation)
├── run_semantic.py      # Multi-seed execution for Semantic Baseline
├── run_topological.py   # Multi-seed execution for Topological Baseline
├── run_hybrid.py        # Multi-seed execution for Hybrid Model & Continual Learning
└── run_ablation.py      # Multi-seed execution for Structural Ablation Study
```

---

## Prerequisites

The framework requires **Python 3.10+** and a **CUDA-enabled GPU** for optimal performance. Key dependencies include:

* `torch >= 2.0.0`
* `torch_geometric >= 2.3.0`
* `pandas >= 2.0.0`
* `numpy >= 1.24.0`
* `scikit-learn >= 1.2.0`
* `matplotlib >= 3.7.0`

---

## Usage and Reproducibility

To ensure strict statistical rigor, all execution scripts (`run_*.py`) automatically iterate over 5 predefined random seeds (`[42, 123, 777, 1024, 2026]`), aggregating the results and computing the final **Mean ± Standard Deviation** for all classification metrics.

### 1. Data Preparation
Run the preprocessing scripts in sequence to build the vectorized datasets and PyG graph structures from the raw historical (PHEME) and contemporary (USE24-XD) corpora:

```bash
python preprocessing/01_etl_pipeline.py
python preprocessing/02_feature_extraction_semantic.py
python preprocessing/03_feature_extraction_topological.py
```

### 2. Running the Baselines
Evaluate the isolated modalities under temporal domain shift:

* **Semantic Baseline** (Evaluates purely textual features / Concept Drift vulnerability):
  ```bash
  python run_semantic.py
  ```
* **Topological Baseline** (Evaluates purely structural features / Algorithmic Drift vulnerability):
  ```bash
  python run_topological.py
  ```

### 3. Running the Hybrid Model & Continual Learning
This is the core experiment of the paper. It sequentially performs:
* Historical training on PHEME.
* Naive Fine-Tuning on USE24-XD, evaluated on the USE24-XD test set itself (plasticity baseline) and, upon re-evaluation, on the historical PHEME test set (measuring Catastrophic Forgetting).
* EWC Fine-Tuning on USE24-XD (measuring Memory Retention and Plasticity).
* Fisher Information Matrix tracking by parameter group.
* Gate clamping and classification head refitting to empirically validate branch load-bearing post-EWC.
* EWC Elasticity Parameter Sensitivity Analysis.

```bash
python run_hybrid.py
```

### 4. Running the Ablation Study
To empirically validate the emergent behavior of the adaptive gating mechanism under structural collapse (Algorithmic Drift), run the structural ablation script:

```bash
python run_ablation.py
```

---
## NEW PROTOCOL 20-SEEDS
## Run

**Step 1: smoke test (about 5 minutes).** This checks that the whole pipeline runs, using 1 seed, 1 epoch and a USE24 subsample.

```python
!python run_revision_experiments.py --stage all --quick
```

**Step 2: the real runs, in this order.**

```python
!python run_revision_experiments.py --stage equivalence     # ~1 min
!python run_revision_experiments.py --stage select_lambda   # ~2-3 h
!python run_revision_experiments.py --stage main            # ~4-6 h (20 seeds)
!python run_revision_experiments.py --stage controls        # ~2-3 h
!python run_revision_experiments.py --stage lofo            # ~1 h
!python run_revision_experiments.py --stage summarize       # seconds
```

The durations are estimates for a T4. The first seed's printed time gives you the real figure.

**If Colab disconnects**, just re-run the same command. Every finished unit (one seed, one condition) is saved to Drive immediately and skipped on the next run.

**If time is short**, `--seeds 14` limits `main` and `controls` to the first 14 initialisation seeds. That is the minimum the power analysis in the paper supports. Run `select_lambda` before `main`, because `main` uses the λ chosen there.

## What each stage does

| Stage | Purpose |
|---|---|
| `equivalence` | Transfers weights from the GCN model to the node-wise MLP variant, then compares outputs and gradients on single-node graphs (real USE24 and random inputs). Multi-node PHEME graphs are included as a control where the outputs must differ. |
| `select_lambda` | Chooses λ using **validation data only**: PHEME validation for retention, and a 10% USE24 validation split for plasticity. It uses a fixed split and 5 seeds, and a rule fixed in advance (see the code). No test set is used. |
| `main` | Confirmatory runs on a **fixed split with 20 initialisation seeds**. For each seed: naive fine-tuning, EWC(λ\*), and uniform (L2-SP-like) anchoring (plus EWC at 50,000 if λ\* differs). It also records the gate trajectory per epoch, displacement and realised penalty per parameter group, and clamped-gate results with and without head refit. It computes calibration (ECE, Brier) and keeps the confusion matrices. |
| `controls` | PHEME models trained under **one matched protocol**: graph-only and semantic-only (root-only) variants, edge-removed graphs with the same node features, single-node graphs, static concatenation, and a fixed (input-independent) gate. |
| `lofo` | Leave-one-event-out on PHEME: train on four events and test on the fifth, for the hybrid, graph-only and semantic-only models, with 3 seeds. |
| `summarize` | Writes `summary.json` and `summary.md`: means ± SD, paired t-tests, 95% CIs and d_z. |

## Differences from the original protocol (intentional)

- **Fixed split.** The PHEME and USE24 splits are fixed with `random_state=42`, and only the initialisation seed varies. This is what Reviewer 3 requested.
- **USE24 validation split.** 10% of the USE24 training partition is held out as validation, so fine-tuning uses the remaining 90%.
- **Gradient clipping in the λ search.** The λ search uses the full 5-epoch protocol with clipping.
