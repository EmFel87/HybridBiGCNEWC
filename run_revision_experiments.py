"""Revision experiments for manuscript electronics-4617812.

Place this file in the root of the HybridBiGCNEWC repository (next to
``run_hybrid.py``) and run it on Google Colab with a GPU runtime and the
same Google Drive layout used for the original experiments.

It reuses the repository's own code (data loading, splits, model,
training constants, EWC implementation, evaluation) so that every
protocol detail matches the paper unless a stage explicitly changes it.

STAGES (run in this order; each can be re-run, finished units are skipped)
  equivalence   Weight-transfer check: GCN branch vs node-wise MLP on
                single-node graphs (outputs and gradients).        ~1 min
  select_lambda Choose lambda on held-out VALIDATION data only
                (PHEME validation retention, USE24 validation plasticity),
                fixed split, 5 initialisation seeds.               ~2-3 h
  main          Confirmatory runs: fixed split, 20 initialisation seeds,
                naive / EWC(lambda*) / uniform-anchoring fine-tuning,
                gate trajectories, branch displacement, realised penalty,
                clamped gate with and without head refit, calibration.
                                                                   ~4-6 h
  controls      PHEME-only matched-protocol controls: graph-only and
                semantic-only (root-only) models, edge-removed graphs,
                single-node graphs, static concatenation, fixed gate.
                                                                   ~2-3 h
  lofo          Leave-one-event-out PHEME evaluation.              ~1 h
  summarize     Aggregate everything into summary.json / summary.md.

USAGE (Colab cells)
  from google.colab import drive; drive.mount('/content/drive')
  %cd /content/HybridBiGCNEWC
  !python run_revision_experiments.py --stage all --quick   # smoke test
  !python run_revision_experiments.py --stage equivalence
  !python run_revision_experiments.py --stage select_lambda
  !python run_revision_experiments.py --stage main
  !python run_revision_experiments.py --stage controls
  !python run_revision_experiments.py --stage lofo
  !python run_revision_experiments.py --stage summarize

Send back the folder <DRIVE_BASE>/revision_results (in particular
summary.json and summary.md).

Label encoding follows the repository: 0 = Flagged ("fake" in the code),
1 = Unflagged ("real"). Metric keys such as ``f1_fake`` therefore refer
to the Flagged class.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

import run_hybrid as RH
from models.ablation import HybridGatedMLP
from models.ewc import (
    compute_ewc_penalty,
    compute_fisher_group_stats,
    compute_fisher_matrix,
)
from models.hybrid import HybridGatedBiGCN
from utils import DEVICE, set_seed

# =============================================================================
# Configuration
# =============================================================================


class CFG:
    out_dir = f"{RH.DRIVE_BASE}/revision_results"
    pheme_raw_dir = f"{RH.DRIVE_BASE}/Dataset_2016/pheme-rnr-dataset"

    split_seed = 42                        # fixed split for all new runs
    init_seeds = [42, 123, 777, 1024, 2026,           # original five
                  7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                  43, 47, 53, 59, 61]                  # fifteen more
    select_seeds = [42, 123, 777, 1024, 2026]
    lofo_seeds = [42, 123, 777]
    refit_seeds = 5                        # head refit only for first N seeds

    lambda_grid = [0, 10000, 25000, 50000, 100000]
    plasticity_tolerance = 0.02            # max drop in USE24-val F1 vs lambda=0
    use24_val_fraction = 0.10              # of the USE24 training partition

    epochs_main = RH.EPOCHS_MAIN           # 15
    epochs_ft = RH.EPOCHS_FT               # 5
    refit_epochs = RH.REFIT_EPOCHS         # 3
    alpha_probe_size = 2000                # USE24 test graphs for per-epoch alpha

    quick = False
    use24_subsample = None


def apply_quick_mode() -> None:
    CFG.quick = True
    CFG.out_dir = CFG.out_dir + "_quick"
    CFG.init_seeds = [42]
    CFG.select_seeds = [42]
    CFG.lofo_seeds = [42]
    CFG.refit_seeds = 1
    CFG.lambda_grid = [0, 50000]
    CFG.epochs_main = 1
    CFG.epochs_ft = 1
    CFG.refit_epochs = 1
    CFG.alpha_probe_size = 200
    CFG.use24_subsample = 3000


# =============================================================================
# Persistence (resumable on Colab)
# =============================================================================


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def unit_path(stage: str, key: str) -> str:
    d = os.path.join(CFG.out_dir, stage)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.json")


def is_done(stage: str, key: str) -> bool:
    return os.path.exists(unit_path(stage, key))


def save_unit(stage: str, key: str, payload: dict) -> None:
    tmp = unit_path(stage, key) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(_to_jsonable(payload), fh, indent=1)
    os.replace(tmp, unit_path(stage, key))


def load_stage(stage: str) -> dict:
    d = os.path.join(CFG.out_dir, stage)
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn)) as fh:
                out[fn[:-5]] = json.load(fh)
    return out


# =============================================================================
# Data
# =============================================================================

_DATA_CACHE: dict = {}


def get_data():
    """Load graphs once, using the repository's own loader."""
    if not _DATA_CACHE:
        gp, lp, gu, lu = RH.load_and_build_graphs()
        if CFG.use24_subsample:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(gu), CFG.use24_subsample, replace=False)
            gu = [gu[i] for i in idx]
            lu = [lu[i] for i in idx]
        _DATA_CACHE.update(gp=gp, lp=lp, gu=gu, lu=lu)
    return _DATA_CACHE


def fixed_splits(with_use24_val: bool = True) -> dict:
    """Same split procedure as run_hybrid.py, with a FIXED random_state.

    PHEME: 70/15/15 stratified. USE24: 80/20 stratified; if requested, 10%
    of the USE24 training partition is held out as a validation set (used
    only by select_lambda; the other stages train on the remaining 90%).
    """
    d = get_data()
    s = CFG.split_seed
    tr, tmp, _, tmp_l = train_test_split(
        d["gp"], d["lp"], test_size=0.30, random_state=s, stratify=d["lp"])
    va, te = train_test_split(tmp, test_size=0.50, random_state=s, stratify=tmp_l)
    u_tr, u_te, u_tr_l, _ = train_test_split(
        d["gu"], d["lu"], test_size=0.20, random_state=s, stratify=d["lu"])
    u_va = []
    if with_use24_val:
        u_tr, u_va = train_test_split(
            u_tr, test_size=CFG.use24_val_fraction, random_state=s,
            stratify=u_tr_l)
    return dict(ph_train=tr, ph_val=va, ph_test=te,
                u_train=u_tr, u_val=u_va, u_test=u_te)


def loader(graphs, shuffle=False):
    return DataLoader(graphs, batch_size=RH.BATCH_SIZE, shuffle=shuffle)


def strip_edges(graphs):
    """Same nodes and node features, no edges."""
    out = []
    for g in graphs:
        h = g.clone()
        h.edge_index = torch.empty((2, 0), dtype=torch.long)
        out.append(h)
    return out


def to_singleton(graphs):
    """Root node only (index 0 = smallest tweet id = source post)."""
    out = []
    for g in graphs:
        h = g.clone()
        h.x = g.x[:1].clone()
        h.edge_index = torch.empty((2, 0), dtype=torch.long)
        h.num_nodes = 1
        out.append(h)
    return out


# =============================================================================
# Model variants (fixed-fusion controls)
# =============================================================================


class _Branches(HybridGatedBiGCN):
    """Exact copy of HybridGatedBiGCN's branch computation."""

    def branches(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index_bu = edge_index.flip(0)
        x_td = F.gelu(self.td_conv1(x, edge_index))
        x_td = F.gelu(self.td_conv2(x_td, edge_index))
        g_td = global_mean_pool(x_td, batch)
        x_bu = F.gelu(self.bu_conv1(x, edge_index_bu))
        x_bu = F.gelu(self.bu_conv2(x_bu, edge_index_bu))
        g_bu = global_mean_pool(x_bu, batch)
        z_topo = self.graph_proj(torch.cat([g_td, g_bu], dim=1))
        z_sem = self.text_proj(x[data.ptr[:-1]])
        return z_sem, z_topo


class ConcatBiGCN(_Branches):
    """Static concatenation [z_sem || z_topo] -> head (no gate)."""

    def __init__(self, common_dim: int = 128):
        super().__init__(common_dim=common_dim)
        del self.gate
        self.fc1 = nn.Linear(common_dim * 2, 64)

    def forward(self, data, force_alpha=None):
        z_sem, z_topo = self.branches(data)
        h = torch.cat([z_sem, z_topo], dim=1)
        return self.fc2(self.dropout(F.gelu(self.fc1(h))))


class FixedGateBiGCN(_Branches):
    """Learned but input-independent per-dimension gate alpha = sigmoid(w)."""

    def __init__(self, common_dim: int = 128):
        super().__init__(common_dim=common_dim)
        del self.gate
        self.alpha_logit = nn.Parameter(torch.zeros(common_dim))

    def forward(self, data, force_alpha=None):
        z_sem, z_topo = self.branches(data)
        alpha = torch.sigmoid(self.alpha_logit).unsqueeze(0).expand_as(z_sem)
        self.last_alpha = alpha.detach()
        h = alpha * z_sem + (1.0 - alpha) * z_topo
        return self.fc2(self.dropout(F.gelu(self.fc1(h))))


# =============================================================================
# Training / evaluation helpers (same protocol as run_hybrid.py)
# =============================================================================


def make_criterion():
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.5], device=DEVICE))


def forward(model, batch, force_alpha):
    if force_alpha is None:
        return model(batch)
    return model(batch, force_alpha=force_alpha)


def train(model, train_graphs, epochs, criterion, force_alpha=None,
          fisher=None, opt_params=None, lam=0.0, val_graphs=None,
          epoch_callback=None):
    """AdamW, lr/wd/clip/batch as in run_hybrid.py; optional EWC penalty."""
    opt = optim.AdamW(model.parameters(), lr=RH.LEARNING_RATE,
                      weight_decay=RH.WEIGHT_DECAY)
    tl = loader(train_graphs, shuffle=True)
    vl = loader(val_graphs) if val_graphs else None
    hist = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for b in tl:
            b = b.to(DEVICE)
            opt.zero_grad()
            loss = criterion(forward(model, b, force_alpha).view(-1), b.y.view(-1))
            if fisher is not None and lam > 0:
                loss = loss + compute_ewc_penalty(
                    model=model, fisher_dict=fisher, opt_params=opt_params,
                    lambda_val=lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), RH.CLIP_NORM)
            opt.step()
            tot += loss.item()
        hist["train_loss"].append(tot / max(1, len(tl)))
        if vl is not None:
            model.eval()
            v = 0.0
            with torch.no_grad():
                for b in vl:
                    b = b.to(DEVICE)
                    v += criterion(forward(model, b, force_alpha).view(-1),
                                   b.y.view(-1)).item()
            hist["val_loss"].append(v / max(1, len(vl)))
        if epoch_callback is not None:
            epoch_callback(ep, model)
    return hist


def evaluate(model, graphs, force_alpha=None):
    return RH.evaluate_model(model, loader(graphs), DEVICE, force_alpha=force_alpha)


def mean_alpha(model, graphs):
    return RH.get_alpha(model, loader(graphs), DEVICE)


def calibration(model, graphs, n_bins=10):
    """ECE (on predicted-class confidence) and Brier score for P(y=1)."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for b in loader(graphs):
            b = b.to(DEVICE)
            probs.extend(torch.sigmoid(model(b).view(-1)).cpu().tolist())
            labels.extend(b.y.view(-1).cpu().tolist())
    p = np.asarray(probs)
    y = np.asarray(labels)
    pred = (p > 0.5).astype(float)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return {"ece": float(ece), "brier": float(np.mean((p - y) ** 2))}


def group_of(name: str):
    for g, prefixes in RH.FISHER_PARAM_GROUPS.items():
        if any(pref in name for pref in prefixes):
            return g
    return None


def displacement_and_penalty(model, opt_params, fisher, lam):
    """Per-group squared displacement from theta*_A and realised penalty."""
    out = {}
    for name, p in model.named_parameters():
        if name not in opt_params:
            continue
        g = group_of(name)
        if g is None:
            continue
        d2 = (p.detach() - opt_params[name]).pow(2)
        rec = out.setdefault(g, {"sum_sq_disp": 0.0, "n": 0, "penalty": 0.0})
        rec["sum_sq_disp"] += float(d2.sum())
        rec["n"] += int(d2.numel())
        rec["penalty"] += float((lam / 2.0) * (fisher[name] * d2).sum())
    for rec in out.values():
        rec["mean_sq_disp"] = rec["sum_sq_disp"] / max(1, rec["n"])
    return out


def uniform_fisher(fisher):
    """Uniform anchoring (L2-SP-like) with the same total Fisher mass."""
    total = sum(float(v.sum()) for v in fisher.values())
    count = sum(v.numel() for v in fisher.values())
    c = total / count
    return {k: torch.full_like(v, c) for k, v in fisher.items()}


def train_hybrid_on_pheme(sp, seed, criterion):
    set_seed(seed)
    model = HybridGatedBiGCN().to(DEVICE)
    hist = train(model, sp["ph_train"], CFG.epochs_main, criterion,
                 val_graphs=sp["ph_val"])
    return model, hist


# =============================================================================
# Stage 0: implementation equivalence (GCN branch vs MLP on singletons)
# =============================================================================


def stage_equivalence():
    if is_done("equivalence", "result"):
        print("[equivalence] already done")
        return
    torch.manual_seed(0)
    gcn = HybridGatedBiGCN().to(DEVICE)
    mlp = HybridGatedMLP().to(DEVICE)
    sd = gcn.state_dict()
    mapped = {}
    for k, v in sd.items():
        k2 = k
        for conv, lin in [("td_conv1", "td_lin1"), ("td_conv2", "td_lin2"),
                          ("bu_conv1", "bu_lin1"), ("bu_conv2", "bu_lin2")]:
            if k.startswith(conv + ".lin."):
                k2 = k.replace(conv + ".lin.", lin + ".")
            elif k == conv + ".bias":
                k2 = lin + ".bias"
        mapped[k2] = v
    missing, unexpected = mlp.load_state_dict(mapped, strict=False)
    report = {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}

    d = get_data()
    real = d["gu"][:256]                      # real USE24 single-node graphs
    rand = []
    for g in real[:64]:
        h = g.clone()
        h.x = torch.randn_like(g.x)
        rand.append(h)
    multi = [g for g in d["gp"] if g.num_nodes > 3][:64]   # control: real cascades

    criterion = make_criterion()

    def compare(graphs, tag):
        b = next(iter(DataLoader(graphs, batch_size=len(graphs)))).to(DEVICE)
        gcn.eval(); mlp.eval()
        gcn.zero_grad(); mlp.zero_grad()
        o1 = gcn(b).view(-1); o2 = mlp(b).view(-1)
        criterion(o1, b.y.view(-1)).backward()
        criterion(o2, b.y.view(-1)).backward()
        g1 = dict(gcn.named_parameters()); g2 = dict(mlp.named_parameters())
        max_grad = 0.0
        for k1, k2 in zip(sd.keys(), mapped.keys()):
            if k1 in g1 and k2 in g2 and g1[k1].grad is not None and g2[k2].grad is not None:
                max_grad = max(max_grad, float((g1[k1].grad - g2[k2].grad).abs().max()))
        report[tag] = {"n_graphs": len(graphs),
                       "max_abs_output_diff": float((o1 - o2).abs().max()),
                       "max_abs_grad_diff": max_grad}

    compare(real, "use24_real_singletons")
    compare(rand, "random_singletons")
    compare(multi, "pheme_multinode_control")
    save_unit("equivalence", "result", report)
    print("[equivalence]", json.dumps(report, indent=1))


# =============================================================================
# Stage 1: lambda selection on VALIDATION data only
# =============================================================================


def stage_select_lambda():
    sp = fixed_splits(with_use24_val=True)
    criterion = make_criterion()
    for seed in CFG.select_seeds:
        key = f"seed{seed}"
        if is_done("select_lambda", key):
            print(f"[select_lambda] {key} done"); continue
        t0 = time.time()
        base, _ = train_hybrid_on_pheme(sp, seed, criterion)
        fisher, opt_params = compute_fisher_matrix(
            base, loader(sp["ph_train"], shuffle=True), criterion, DEVICE)
        res = {}
        for lam in CFG.lambda_grid:
            set_seed(seed)
            m = copy.deepcopy(base)
            train(m, sp["u_train"], CFG.epochs_ft, criterion,
                  fisher=fisher, opt_params=opt_params, lam=lam)
            res[str(lam)] = {
                "pheme_val": evaluate(m, sp["ph_val"]),
                "use24_val": evaluate(m, sp["u_val"]),
            }
        save_unit("select_lambda", key, {"seed": seed, "by_lambda": res,
                                         "minutes": (time.time() - t0) / 60})
        print(f"[select_lambda] {key} saved ({(time.time()-t0)/60:.1f} min)")
    choose_lambda(write=True)


def choose_lambda(write=False) -> int:
    """Pre-specified rule: among lambda > 0 whose mean USE24-validation
    F1-Flagged is within `plasticity_tolerance` of lambda = 0, choose the
    one with the highest mean PHEME-validation F1-Flagged (retention);
    ties -> smaller lambda. Falls back to 50,000 if no data."""
    runs = load_stage("select_lambda")
    runs = {k: v for k, v in runs.items() if k.startswith("seed")}
    if not runs:
        return RH.LAMBDA_EWC
    lams = [int(l) for l in next(iter(runs.values()))["by_lambda"].keys()]
    table = {}
    for lam in lams:
        ret = [r["by_lambda"][str(lam)]["pheme_val"]["f1_fake"] for r in runs.values()]
        pla = [r["by_lambda"][str(lam)]["use24_val"]["f1_fake"] for r in runs.values()]
        table[lam] = {"retention_val_mean": float(np.mean(ret)),
                      "plasticity_val_mean": float(np.mean(pla))}
    ref = table.get(0, {}).get("plasticity_val_mean", -1)
    ok = [l for l in lams if l > 0 and table[l]["plasticity_val_mean"] >= ref - CFG.plasticity_tolerance]
    chosen = max(sorted(ok), key=lambda l: table[l]["retention_val_mean"]) if ok else RH.LAMBDA_EWC
    if write:
        save_unit("select_lambda", "selection",
                  {"rule": choose_lambda.__doc__, "table": table,
                   "chosen_lambda": chosen, "n_seeds": len(runs)})
        print(f"[select_lambda] chosen lambda = {chosen}")
    return chosen


# =============================================================================
# Stage 2: confirmatory multi-seed runs (fixed split, untouched test sets)
# =============================================================================


def stage_main():
    lam_star = choose_lambda()
    sp = fixed_splits(with_use24_val=True)
    criterion = make_criterion()
    probe = sp["u_test"][:CFG.alpha_probe_size]
    variants = {"naive": 0}
    variants[f"ewc_{lam_star}"] = lam_star
    if lam_star != RH.LAMBDA_EWC:
        variants[f"ewc_{RH.LAMBDA_EWC}"] = RH.LAMBDA_EWC
    variants[f"uniform_{lam_star}"] = lam_star

    for i, seed in enumerate(CFG.init_seeds):
        key = f"seed{seed}"
        if is_done("main", key):
            print(f"[main] {key} done"); continue
        t0 = time.time()
        base, hist = train_hybrid_on_pheme(sp, seed, criterion)
        rec = {"seed": seed, "lambda_star": lam_star, "loss_history": hist,
               "pheme_base": evaluate(base, sp["ph_test"]),
               "use24_cold": evaluate(base, sp["u_test"]),
               "alpha_pheme": mean_alpha(base, sp["ph_test"]),
               "alpha_use24_cold": mean_alpha(base, sp["u_test"]),
               "calibration_pheme_base": calibration(base, sp["ph_test"])}
        fisher, opt_params = compute_fisher_matrix(
            base, loader(sp["ph_train"], shuffle=True), criterion, DEVICE)
        rec["fisher_stats"] = compute_fisher_group_stats(fisher, RH.FISHER_PARAM_GROUPS)
        fisher_u = uniform_fisher(fisher)

        for vname, lam in variants.items():
            set_seed(seed)
            m = copy.deepcopy(base)
            traj = []
            fdict = fisher_u if vname.startswith("uniform") else fisher
            train(m, sp["u_train"], CFG.epochs_ft, criterion,
                  fisher=fdict, opt_params=opt_params, lam=lam,
                  epoch_callback=lambda ep, mm: traj.append(mean_alpha(mm, probe)))
            v = {"lambda": lam,
                 "pheme_test": evaluate(m, sp["ph_test"]),
                 "use24_test": evaluate(m, sp["u_test"]),
                 "alpha_use24_test": mean_alpha(m, sp["u_test"]),
                 "alpha_pheme_test": mean_alpha(m, sp["ph_test"]),
                 "alpha_trajectory_use24_probe": traj,
                 "displacement_fisher_penalty": displacement_and_penalty(
                     m, opt_params, fisher, lam_star),
                 "calibration_use24": calibration(m, sp["u_test"])}
            if vname.startswith("uniform"):
                v["displacement_uniform_penalty"] = displacement_and_penalty(
                    m, opt_params, fisher_u, lam)
            if vname == f"ewc_{lam_star}":
                v["clamp_alpha1_norefit"] = evaluate(m, sp["u_test"], force_alpha=1.0)
                v["clamp_alpha0_norefit"] = evaluate(m, sp["u_test"], force_alpha=0.0)
                if i < CFG.refit_seeds:
                    set_seed(seed)
                    v["clamp_alpha1_refit"] = RH.refit_head(
                        m, loader(sp["u_train"], shuffle=True), loader(sp["u_test"]),
                        criterion, DEVICE, force_alpha=1.0, epochs=CFG.refit_epochs)
                    set_seed(seed)
                    v["clamp_alpha0_refit"] = RH.refit_head(
                        m, loader(sp["u_train"], shuffle=True), loader(sp["u_test"]),
                        criterion, DEVICE, force_alpha=0.0, epochs=CFG.refit_epochs)
            rec[vname] = v
        rec["minutes"] = (time.time() - t0) / 60
        save_unit("main", key, rec)
        print(f"[main] {key} saved ({rec['minutes']:.1f} min)")


# =============================================================================
# Stage 3: PHEME matched-protocol controls
# =============================================================================

CONTROL_CONDITIONS = {
    # name: (model factory, force_alpha during train/eval, graph transform)
    "hybrid_intact":        (HybridGatedBiGCN, None, None),
    "hybrid_edges_removed": (HybridGatedBiGCN, None, "strip"),
    "hybrid_singleton":     (HybridGatedBiGCN, None, "singleton"),
    "graph_only_intact":    (HybridGatedBiGCN, 0.0,  None),
    "graph_only_edges_removed": (HybridGatedBiGCN, 0.0, "strip"),   # = graph-free aggregation of all post embeddings
    "graph_only_singleton": (HybridGatedBiGCN, 0.0,  "singleton"),
    "semantic_only_root":   (HybridGatedBiGCN, 1.0,  None),        # root-only thread-level model
    "static_concat":        (ConcatBiGCN, None, None),
    "fixed_gate":           (FixedGateBiGCN, None, None),
}


def stage_controls():
    sp = fixed_splits(with_use24_val=True)
    criterion = make_criterion()
    transformed = {None: sp,
                   "strip": {k: strip_edges(sp[k]) for k in ("ph_train", "ph_val", "ph_test")},
                   "singleton": {k: to_singleton(sp[k]) for k in ("ph_train", "ph_val", "ph_test")}}
    for seed in CFG.init_seeds:
        for cname, (factory, fa, tf) in CONTROL_CONDITIONS.items():
            key = f"{cname}__seed{seed}"
            if is_done("controls", key):
                continue
            t0 = time.time()
            data = transformed[tf]
            set_seed(seed)
            m = factory().to(DEVICE)
            train(m, data["ph_train"], CFG.epochs_main, criterion,
                  force_alpha=fa, val_graphs=data["ph_val"])
            rec = {"seed": seed, "condition": cname,
                   "pheme_test": evaluate(m, data["ph_test"], force_alpha=fa)}
            if cname in ("hybrid_intact", "fixed_gate"):
                rec["alpha_pheme_test"] = mean_alpha(m, data["ph_test"])
            rec["minutes"] = (time.time() - t0) / 60
            save_unit("controls", key, rec)
            print(f"[controls] {key}: F1-Flagged {rec['pheme_test']['f1_fake']:.4f}")


# =============================================================================
# Stage 4: leave-one-event-out PHEME
# =============================================================================


def build_event_map() -> dict:
    """root_id -> event, from the raw PHEME folder structure
    <root>/<event>/<rumours|non-rumours>/<root_id>/..."""
    m = {}
    for ev in sorted(os.listdir(CFG.pheme_raw_dir)):
        ev_dir = os.path.join(CFG.pheme_raw_dir, ev)
        if not os.path.isdir(ev_dir):
            continue
        for cls in ("rumours", "non-rumours"):
            cdir = os.path.join(ev_dir, cls)
            if not os.path.isdir(cdir):
                continue
            for tid in os.listdir(cdir):
                if not tid.startswith("."):
                    m[str(tid)] = ev
    return m


def stage_lofo():
    d = get_data()
    ev_map = build_event_map()
    graphs = [g for g in d["gp"] if str(g.root_id) in ev_map]
    print(f"[lofo] {len(graphs)} / {len(d['gp'])} PHEME threads mapped to events")
    events = sorted({ev_map[str(g.root_id)] for g in graphs})
    if CFG.quick:
        events = events[:1]
    criterion = make_criterion()
    conds = {"hybrid": None, "graph_only": 0.0, "semantic_only_root": 1.0}
    for ev in events:
        test = [g for g in graphs if ev_map[str(g.root_id)] == ev]
        pool = [g for g in graphs if ev_map[str(g.root_id)] != ev]
        pool_l = [int(g.y.item()) for g in pool]
        for seed in CFG.lofo_seeds:
            tr, va = train_test_split(pool, test_size=0.15, random_state=seed,
                                      stratify=pool_l)
            for cname, fa in conds.items():
                key = f"{ev}__{cname}__seed{seed}"
                if is_done("lofo", key):
                    continue
                set_seed(seed)
                m = HybridGatedBiGCN().to(DEVICE)
                train(m, tr, CFG.epochs_main, criterion, force_alpha=fa, val_graphs=va)
                rec = {"event": ev, "condition": cname, "seed": seed,
                       "n_test": len(test),
                       "flagged_share_test": float(np.mean([g.y.item() == 0 for g in test])),
                       "pheme_heldout": evaluate(m, test, force_alpha=fa)}
                save_unit("lofo", key, rec)
                print(f"[lofo] {key}: F1-Flagged {rec['pheme_heldout']['f1_fake']:.4f}")


# =============================================================================
# Stage 5: summary
# =============================================================================

METRICS = ["acc", "p_fake", "p_real", "r_fake", "r_real", "f1_fake", "f1_real", "f1_macro"]


def ms(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd_pop": float(a.std(ddof=0)),
            "sd_sample": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": int(len(a))}


def paired(a, b):
    """a - b, paired by seed: mean, 95% t-CI, d_z, t, p."""
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    n = len(d)
    if n < 2:
        return {"mean_diff": float(d.mean()) if n else None, "n": n}
    sd = d.std(ddof=1)
    se = sd / math.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    out = {"mean_diff": float(d.mean()), "ci95": [float(d.mean() - tc * se), float(d.mean() + tc * se)],
           "n": n, "n_positive": int((d > 0).sum()), "d_z": None, "t": None, "p": None, "r": None}
    if sd > 0:
        t, p = stats.ttest_rel(a, b)
        out.update(d_z=float(d.mean() / sd), t=float(t), p=float(p))
    if n > 2 and a.std() > 0 and b.std() > 0:
        out["r"] = float(np.corrcoef(a, b)[0, 1])
    return out


def stage_summarize():
    S = {}
    eq = load_stage("equivalence")
    if eq:
        S["equivalence"] = eq.get("result")
    sel = load_stage("select_lambda")
    if "selection" in sel:
        S["lambda_selection"] = sel["selection"]

    main = {k: v for k, v in load_stage("main").items()}
    if main:
        runs = sorted(main.values(), key=lambda r: r["seed"])
        lam = runs[0]["lambda_star"]
        S["main"] = {"seeds": [r["seed"] for r in runs], "lambda_star": lam}
        blocks = {"pheme_base": [r["pheme_base"] for r in runs],
                  "use24_cold": [r["use24_cold"] for r in runs]}
        vnames = [k for k in runs[0] if isinstance(runs[0][k], dict) and "pheme_test" in runs[0][k]]
        for vn in vnames:
            blocks[f"{vn}__pheme"] = [r[vn]["pheme_test"] for r in runs]
            blocks[f"{vn}__use24"] = [r[vn]["use24_test"] for r in runs]
        S["main"]["metrics"] = {b: {m: ms([x[m] for x in lst]) for m in METRICS}
                                for b, lst in blocks.items()}
        S["main"]["alpha"] = {"pheme": ms([r["alpha_pheme"] for r in runs]),
                              "use24_cold": ms([r["alpha_use24_cold"] for r in runs])}
        for vn in vnames:
            S["main"]["alpha"][f"{vn}__use24"] = ms([r[vn]["alpha_use24_test"] for r in runs])
            S["main"]["alpha"][f"{vn}__trajectory_mean"] = np.mean(
                [r[vn]["alpha_trajectory_use24_probe"] for r in runs], axis=0).tolist()
        ewc = f"ewc_{lam}"; uni = f"uniform_{lam}"
        f = lambda v, dom, m="f1_fake": [r[v][dom][m] for r in runs]
        S["main"]["paired"] = {
            "retention_f1_ewc_vs_naive": paired(f(ewc, "pheme_test"), f("naive", "pheme_test")),
            "retention_f1_ewc_vs_uniform": paired(f(ewc, "pheme_test"), f(uni, "pheme_test")),
            "retention_f1_uniform_vs_naive": paired(f(uni, "pheme_test"), f("naive", "pheme_test")),
            "use24_f1_ewc_vs_naive": paired(f(ewc, "use24_test"), f("naive", "use24_test")),
            "use24_macro_ewc_vs_naive": paired(f(ewc, "use24_test", "f1_macro"), f("naive", "use24_test", "f1_macro")),
            "alpha_use24_ewc_vs_naive": paired([r[ewc]["alpha_use24_test"] for r in runs],
                                               [r["naive"]["alpha_use24_test"] for r in runs]),
            "clamp_norefit_alpha0_vs_alpha1": paired(
                [r[ewc]["clamp_alpha0_norefit"]["f1_fake"] for r in runs],
                [r[ewc]["clamp_alpha1_norefit"]["f1_fake"] for r in runs]),
        }
        rr = [r for r in runs if "clamp_alpha0_refit" in r[ewc]]
        if len(rr) > 1:
            S["main"]["paired"]["clamp_refit_alpha0_vs_alpha1"] = paired(
                [r[ewc]["clamp_alpha0_refit"]["f1_fake"] for r in rr],
                [r[ewc]["clamp_alpha1_refit"]["f1_fake"] for r in rr])
        S["main"]["clamp"] = {k: ms([r[ewc][k]["f1_fake"] for r in runs if k in r[ewc]])
                              for k in ("clamp_alpha1_norefit", "clamp_alpha0_norefit",
                                        "clamp_alpha1_refit", "clamp_alpha0_refit")}
        groups = list(runs[0]["fisher_stats"].keys())
        S["main"]["fisher"] = {g: {s: ms([r["fisher_stats"][g][s] for r in runs])
                                   for s in ("mean", "median")} for g in groups}
        S["main"]["displacement"] = {
            vn: {g: {"mean_sq_disp": ms([r[vn]["displacement_fisher_penalty"][g]["mean_sq_disp"] for r in runs]),
                     "fisher_penalty": ms([r[vn]["displacement_fisher_penalty"][g]["penalty"] for r in runs])}
                 for g in runs[0][vn]["displacement_fisher_penalty"]}
            for vn in vnames}
        S["main"]["calibration"] = {vn: {c: ms([r[vn]["calibration_use24"][c] for r in runs])
                                         for c in ("ece", "brier")} for vn in vnames}

    ctrl = load_stage("controls")
    if ctrl:
        by = {}
        for v in ctrl.values():
            by.setdefault(v["condition"], {})[v["seed"]] = v["pheme_test"]
        seeds = sorted(set.intersection(*[set(d.keys()) for d in by.values()]))
        S["controls"] = {"seeds": seeds,
                         "metrics": {c: {m: ms([by[c][s][m] for s in seeds]) for m in METRICS}
                                     for c in by}}
        pf = lambda c, m="f1_fake": [by[c][s][m] for s in seeds]
        comps = [("hybrid_intact", "graph_only_intact"),
                 ("hybrid_intact", "semantic_only_root"),
                 ("graph_only_intact", "graph_only_edges_removed"),   # reply structure
                 ("graph_only_edges_removed", "semantic_only_root"),  # reply content
                 ("graph_only_edges_removed", "graph_only_singleton"),
                 ("hybrid_intact", "hybrid_edges_removed"),
                 ("hybrid_intact", "hybrid_singleton"),
                 ("hybrid_intact", "static_concat"),                  # gate vs fixed fusion
                 ("hybrid_intact", "fixed_gate")]
        S["controls"]["paired_f1_flagged"] = {f"{a}__vs__{b}": paired(pf(a), pf(b))
                                              for a, b in comps if a in by and b in by}
        S["controls"]["paired_macro_f1"] = {f"{a}__vs__{b}": paired(pf(a, "f1_macro"), pf(b, "f1_macro"))
                                            for a, b in comps if a in by and b in by}

    lofo = load_stage("lofo")
    if lofo:
        agg = {}
        for v in lofo.values():
            agg.setdefault(v["event"], {}).setdefault(v["condition"], []).append(v["pheme_heldout"])
        S["lofo"] = {ev: {c: {m: ms([x[m] for x in lst]) for m in ("f1_fake", "f1_macro", "acc")}
                          for c, lst in conds.items()} for ev, conds in agg.items()}
        allc = {}
        for ev, conds in agg.items():
            for c, lst in conds.items():
                allc.setdefault(c, []).append(np.mean([x["f1_fake"] for x in lst]))
        S["lofo_macro_average_over_events"] = {c: ms(v) for c, v in allc.items()}

    os.makedirs(CFG.out_dir, exist_ok=True)
    with open(os.path.join(CFG.out_dir, "summary.json"), "w") as fh:
        json.dump(_to_jsonable(S), fh, indent=1)
    write_markdown(S)
    print(f"[summarize] written to {CFG.out_dir}/summary.json and summary.md")


def write_markdown(S):
    L = ["# Revision experiments: summary", ""]
    fmt = lambda d: f"{d['mean']:.4f} ± {d['sd_pop']:.4f}"
    if S.get("equivalence"):
        L += ["## Equivalence check", "```", json.dumps(S["equivalence"], indent=1), "```", ""]
    if S.get("lambda_selection"):
        L += ["## Lambda selection (validation only)", "```",
              json.dumps(S["lambda_selection"], indent=1), "```", ""]
    if S.get("main"):
        M = S["main"]
        L += [f"## Main (fixed split, n = {len(M['seeds'])} seeds, lambda* = {M['lambda_star']})", "",
              "| block | Acc | F1-Flagged | F1-Unflagged | Macro-F1 | R-Flagged | R-Unflagged |",
              "|---|---|---|---|---|---|---|"]
        for b, d in M["metrics"].items():
            L.append(f"| {b} | {fmt(d['acc'])} | {fmt(d['f1_fake'])} | {fmt(d['f1_real'])} | "
                     f"{fmt(d['f1_macro'])} | {fmt(d['r_fake'])} | {fmt(d['r_real'])} |")
        L += ["", "### Paired comparisons", "```", json.dumps(M["paired"], indent=1), "```",
              "### Gate", "```", json.dumps(M["alpha"], indent=1), "```",
              "### Clamp", "```", json.dumps(M["clamp"], indent=1), "```",
              "### Fisher / displacement / penalty", "```",
              json.dumps({"fisher": M["fisher"], "displacement": M["displacement"]}, indent=1), "```",
              "### Calibration (USE24 test)", "```", json.dumps(M["calibration"], indent=1), "```", ""]
    if S.get("controls"):
        C = S["controls"]
        L += [f"## PHEME controls (n = {len(C['seeds'])} seeds)", "",
              "| condition | Acc | F1-Flagged | Macro-F1 |", "|---|---|---|---|"]
        for c, d in C["metrics"].items():
            L.append(f"| {c} | {fmt(d['acc'])} | {fmt(d['f1_fake'])} | {fmt(d['f1_macro'])} |")
        L += ["", "```", json.dumps({"f1_flagged": C["paired_f1_flagged"],
                                      "macro_f1": C["paired_macro_f1"]}, indent=1), "```", ""]
    if S.get("lofo"):
        L += ["## Leave-one-event-out", "```", json.dumps(S["lofo"], indent=1), "```",
              "```", json.dumps(S["lofo_macro_average_over_events"], indent=1), "```"]
    with open(os.path.join(CFG.out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(L))


# =============================================================================
# Entry point
# =============================================================================

STAGES = {"equivalence": stage_equivalence, "select_lambda": stage_select_lambda,
          "main": stage_main, "controls": stage_controls, "lofo": stage_lofo,
          "summarize": stage_summarize}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    ap.add_argument("--quick", action="store_true",
                    help="1 seed, 1 epoch, USE24 subsample: pipeline smoke test")
    ap.add_argument("--seeds", type=int, default=None,
                    help="use only the first N initialisation seeds in main/controls")
    args = ap.parse_args()
    if args.quick:
        apply_quick_mode()
    if args.seeds:
        CFG.init_seeds = CFG.init_seeds[:args.seeds]
    print(f"Device: {DEVICE} | output: {CFG.out_dir} | quick={CFG.quick}")
    order = list(STAGES) if args.stage == "all" else [args.stage]
    for st in order:
        t0 = time.time()
        print(f"\n===== stage: {st} =====")
        STAGES[st]()
        print(f"===== {st} finished in {(time.time()-t0)/60:.1f} min =====")
