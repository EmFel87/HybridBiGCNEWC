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
│   ├── ewc.py           # Elastic Weight Consolidation (Fisher Matrix & Penalty)
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
* Naive Fine-Tuning on USE24-XD (measuring Catastrophic Forgetting).
* EWC Fine-Tuning on USE24-XD (measuring Memory Retention and Plasticity).
* EWC Elasticity Parameter Sensitivity Analysis.

```bash
python run_hybrid.py
```

### 4. Running the Ablation Study
To empirically validate the emergent behavior of the adaptive gating mechanism under structural collapse (Algorithmic Drift), run the structural ablation script:

```bash
python run_ablation.py
```
