# =============================================================================
# FILE:    run_ablation.py
# SCOPO:   Script di esecuzione per l'Ablation Study del progetto
#          "Fake News Detection: 2016 vs 2024". Valida empiricamente il
#          contributo del message passing topologico nel modello
#          ibrido: HybridGatedMLP condivide con HybridGatedBiGCN tutte
#          le componenti (proiezioni, gate, classificatore) ma sostitu-
#          isce le convoluzioni grafiche con trasformazioni lineari per
#          nodo, ignorando la struttura degli archi.
#          Per ciascun seed del protocollo multi-run:
#            1. Training storico dell'MLP su PHEME (15 epoche, nessuna
#               validazione, nessun gradient clipping — fedele al
#               notebook originale).
#            2. Estrazione della Fisher Information Matrix sul training
#               set di PHEME.
#            3. Fine-tuning EWC su USE24 (5 epoche, lambda=50000).
#            4. Valutazione finale sul test set di USE24.
#          Se la F1-Fake ottenuta qui e' significativamente inferiore a
#          quella di HybridGatedBiGCN post-EWC (run_hybrid.py), la
#          differenza e' attribuibile esclusivamente alla capacita' di
#          sfruttare la topologia della cascata.
#          L'architettura e' importata da models.ablation, la logica
#          EWC da models.ewc, le utility di riproducibilita', metriche
#          e aggregazione dei risultati da utils.
# DIPENDENZE: torch, torch_geometric, pandas, numpy, scikit-learn
# MODULO:  Esecuzione — Ablation Study (Gated MLP, senza message passing)
# =============================================================================


# ##############################################################################
# ### FASE 0: IMPORT E COSTANTI GLOBALI                                      ###
# ##############################################################################

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# NOTA: DataLoader di PyTorch Geometric, non quello standard di
# torch.utils.data — necessario per il collate corretto dei batch di
# grafi (super-grafo sparso a blocchi diagonali).
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from models.ablation import HybridGatedMLP
from models.ewc import compute_fisher_matrix, compute_ewc_penalty
from utils import (
    DEVICE,
    set_seed,
    compute_classification_metrics,
    init_results_dict,
    append_run_results,
    print_stats,
)

# -- Percorsi Google Drive ----------------------------------------------------
DRIVE_BASE       = "/content/drive/MyDrive/Tesi"
IN_PHEME_PARQUET = f"{DRIVE_BASE}/Parquet_Finali/PHEME_Vectorized_FP16.parquet"
IN_PHEME_GRAPHS  = f"{DRIVE_BASE}/Grafi_PyG/PHEME_graphs_list.pt"
IN_USE24_PARQUET = f"{DRIVE_BASE}/Parquet_Finali/USE24_Vectorized_FP16.parquet"
IN_USE24_GRAPHS  = f"{DRIVE_BASE}/Grafi_PyG/USE24_graphs_list.pt"

# -- Mapping label testuale -> classe binaria (0 = Fake, 1 = Real) -----------
PHEME_LABEL_MAP: dict[str, int] = {
    "rumour":     0,
    "rumor":      0,
    "non-rumour": 1,
    "non-rumor":  1,
}

USE24_LABEL_MAP: dict[str, int] = {
    "neutral":        1,
    "sensationalism": 0,
    "conspiracy":     0,
    "hate_speech":    0,
    "satire":         0,
    "speculation":    0,
}

# -- Protocollo multi-seed -----------------------------------------------------
SEEDS = [42, 123, 777, 1024, 2026]

# -- Iperparametri di training (identici al notebook originale) --------------
BATCH_SIZE     = 32
LEARNING_RATE  = 0.0005
WEIGHT_DECAY   = 1e-4

EPOCHS_MAIN    = 15   # Training storico dell'MLP su PHEME
EPOCHS_FT      = 5    # Fine-tuning EWC su USE24

# -- Coefficiente EWC (identico al fine-tuning principale di run_hybrid.py) --
LAMBDA_EWC     = 50000


# ##############################################################################
# ### FASE 1: CARICAMENTO E COSTRUZIONE DEI GRAFI (ESEGUITO UNA SOLA VOLTA) ###
# ##############################################################################

def load_and_build_graphs() -> tuple[list, list[int], list, list[int]]:
    """Carica i grafi PyG di PHEME e USE24 e inietta le label numeriche.

    Identica alla funzione omonima in ``run_hybrid.py`` e
    ``run_topological.py``: la logica di caricamento e mapping delle
    label e' condivisa da tutti gli esperimenti che operano sui grafi
    di propagazione.

    Returns:
        Tupla ``(valid_graphs_pheme, labels_pheme, valid_graphs_use24,
        labels_use24)`` dove le liste di grafi contengono oggetti
        ``torch_geometric.data.Data`` con ``g.y`` gia' impostato, e le
        liste di label sono le corrispondenti liste parallele di interi
        0/1 usate per lo stratify negli split scikit-learn.
    """
    print("Caricamento dataset PHEME e USE24 (Parquet + Grafi)...")

    # -- 1. PHEME --------------------------------------------------------------
    df_pheme = pd.read_parquet(IN_PHEME_PARQUET)
    target_col_p = "status" if "status" in df_pheme.columns else "label"
    df_pheme["label_num"] = (
        df_pheme[target_col_p]
        .astype(str).str.lower().str.strip()
        .map(PHEME_LABEL_MAP)
    )
    df_pheme = df_pheme.dropna(subset=["label_num"])

    id_col_p = (
        "node_id" if "node_id" in df_pheme.columns else "tweet_id"
    )
    label_dict_pheme: dict[str, int] = dict(
        zip(
            df_pheme[id_col_p].astype(str),
            df_pheme["label_num"].astype(int),
        )
    )

    pheme_graphs_raw = torch.load(IN_PHEME_GRAPHS, weights_only=False)
    valid_graphs_pheme: list      = []
    labels_pheme:       list[int] = []

    for g in pheme_graphs_raw:
        root_id = str(g.root_id)
        if root_id in label_dict_pheme:
            vera_label = label_dict_pheme[root_id]
            g.y = torch.tensor([vera_label], dtype=torch.float32)
            valid_graphs_pheme.append(g)
            labels_pheme.append(vera_label)

    # -- 2. USE24 ----------------------------------------------------------------
    df_use24 = pd.read_parquet(IN_USE24_PARQUET)
    target_col_u = "status" if "status" in df_use24.columns else "label"
    df_use24["label_num"] = (
        df_use24[target_col_u]
        .astype(str).str.lower().str.strip()
        .map(USE24_LABEL_MAP)
    )
    df_use24 = df_use24.dropna(subset=["label_num"])

    id_col_u = (
        "node_id" if "node_id" in df_use24.columns else "tweet_id"
    )
    label_dict_use24: dict[str, int] = dict(
        zip(
            df_use24[id_col_u].astype(str),
            df_use24["label_num"].astype(int),
        )
    )

    use24_graphs_raw = torch.load(IN_USE24_GRAPHS, weights_only=False)
    valid_graphs_use24: list      = []
    labels_use24:       list[int] = []

    for g in use24_graphs_raw:
        root_id = str(g.root_id)
        if root_id in label_dict_use24:
            vera_label = label_dict_use24[root_id]
            g.y = torch.tensor([vera_label], dtype=torch.float32)
            valid_graphs_use24.append(g)
            labels_use24.append(vera_label)

    print(
        f"Dati caricati! PHEME: {len(valid_graphs_pheme)} grafi | "
        f"USE24: {len(valid_graphs_use24)} grafi."
    )

    return (
        valid_graphs_pheme, labels_pheme,
        valid_graphs_use24, labels_use24,
    )


# ##############################################################################
# ### FASE 2: UTILITY DI VALUTAZIONE                                        ###
# ##############################################################################

def evaluate_model(
    model:  HybridGatedMLP,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Esegue l'inferenza su un DataLoader PyG e calcola le metriche complete.

    Args:
        model:  Istanza di ``HybridGatedMLP`` da valutare.
        loader: DataLoader PyG del set di valutazione.
        device: Dispositivo di calcolo.

    Returns:
        Dizionario prodotto da ``compute_classification_metrics``
        (chiavi: ``acc``, ``p_fake``, ``p_real``, ``r_fake``,
        ``r_real``, ``f1_fake``, ``f1_real``, ``f1_macro``, ``cm``).
    """
    model.eval()
    all_preds:  list[float] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            probs = torch.sigmoid(model(batch_data).view(-1))
            all_preds.extend((probs > 0.5).float().cpu().numpy())
            all_labels.extend(batch_data.y.view(-1).cpu().numpy())

    return compute_classification_metrics(all_labels, all_preds)


# ##############################################################################
# ### FASE 3: ESPERIMENTO COMPLETO PER UN SINGOLO SEED                      ###
# ##############################################################################

def run_ablation_for_seed(
    seed:                int,
    valid_graphs_pheme:  list,
    labels_pheme:        list[int],
    valid_graphs_use24:  list,
    labels_use24:        list[int],
    criterion:           nn.Module,
) -> dict:
    """Esegue il ciclo Ablation (MLP) completo per un singolo seed.

    Riproduce esattamente la Fase 5 del notebook originale:
      (A) Split stratificato: 70% di PHEME per il training storico,
          80%/20% di USE24 per fine-tuning/valutazione.
      (B) Training storico: un nuovo ``HybridGatedMLP`` viene
          addestrato da zero su PHEME per ``EPOCHS_MAIN`` epoche.
          A differenza del training di ``HybridGatedBiGCN`` in
          ``run_hybrid.py``, questa fase non include ne' validazione
          ne' gradient clipping, fedelmente al codice originale.
      (C) Fisher Information Matrix: calcolata sul training set di
          PHEME a partire dal modello appena addestrato, tramite
          ``compute_fisher_matrix``.
      (D) EWC Fine-Tuning: lo stesso modello (nessun deepcopy — il
          notebook originale continua ad addestrare ``model_mlp`` in
          place) viene sottoposto a fine-tuning su USE24 con la penale
          EWC per ``EPOCHS_FT`` epoche, anch'esso senza gradient
          clipping.
      (E) Valutazione finale sul test set di USE24.

    Args:
        seed: Seed della run corrente. Deve essere gia' stato
            propagato ai generatori casuali tramite ``set_seed(seed)``
            prima della chiamata.
        valid_graphs_pheme: Lista di grafi PHEME con label gia'
            iniettata, prodotta da ``load_and_build_graphs``.
        labels_pheme: Lista parallela di interi 0/1 per lo stratify
            dello split di PHEME.
        valid_graphs_use24: Lista di grafi USE24 con label gia'
            iniettata, prodotta da ``load_and_build_graphs``.
        labels_use24: Lista parallela di interi 0/1 per lo stratify
            dello split di USE24.
        criterion: Funzione di loss condivisa (``BCEWithLogitsLoss``
            con ``pos_weight``), inizializzata una sola volta fuori
            dal ciclo sui seed.

    Returns:
        Dizionario prodotto da ``compute_classification_metrics`` per
        la valutazione finale sul test set di USE24.
    """
    # -- (A) Split PHEME: solo la partizione di training (70%) e' necessaria --
    train_g, _, _, _ = train_test_split(
        valid_graphs_pheme, labels_pheme,
        test_size=0.30, random_state=seed, stratify=labels_pheme,
    )
    train_loader_ph = DataLoader(
        train_g, batch_size=BATCH_SIZE, shuffle=True
    )

    # -- (A) Split USE24 (80% train fine-tuning, 20% test) ---------------------
    use24_train_g, use24_test_g, _, _ = train_test_split(
        valid_graphs_use24, labels_use24,
        test_size=0.20, random_state=seed, stratify=labels_use24,
    )
    train_loader_u24 = DataLoader(
        use24_train_g, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader_u24 = DataLoader(
        use24_test_g, batch_size=BATCH_SIZE, shuffle=False
    )

    # ------------------------------------------------------------------ #
    # (B) Training storico dell'MLP su PHEME                             #
    # ------------------------------------------------------------------ #
    model_mlp = HybridGatedMLP().to(DEVICE)
    optimizer = optim.AdamW(
        model_mlp.parameters(),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    for _ in range(EPOCHS_MAIN):
        model_mlp.train()
        for batch_data in train_loader_ph:
            batch_data = batch_data.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(
                model_mlp(batch_data).view(-1), batch_data.y.view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_mlp.parameters(), 1.0)
            optimizer.step()

    # ------------------------------------------------------------------ #
    # (C) Estrazione Fisher Information Matrix su PHEME                 #
    # ------------------------------------------------------------------ #
    fisher_dict, opt_params = compute_fisher_matrix(
        model=model_mlp, loader=train_loader_ph,
        criterion=criterion, device=DEVICE,
    )

    # ------------------------------------------------------------------ #
    # (D) EWC Fine-Tuning su USE24                                       #
    # ------------------------------------------------------------------ #
    optimizer_ft = optim.AdamW(
        model_mlp.parameters(),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    model_mlp.train()
    for _ in range(EPOCHS_FT):
        for batch_data in train_loader_u24:
            batch_data = batch_data.to(DEVICE)
            optimizer_ft.zero_grad()

            l_task = criterion(
                model_mlp(batch_data).view(-1), batch_data.y.view(-1)
            )
            # compute_ewc_penalty restituisce gia' il termine scalato
            # per lambda/2: non applicare ulteriori fattori di scala.
            l_ewc = compute_ewc_penalty(
                model=model_mlp,
                fisher_dict=fisher_dict,
                opt_params=opt_params,
                lambda_val=LAMBDA_EWC,
            )
            loss = l_task + l_ewc

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_mlp.parameters(), 1.0)
            optimizer_ft.step()

    # -- (E) Valutazione finale sul test set di USE24 --------------------------
    metrics_use24_mlp = evaluate_model(model_mlp, test_loader_u24, DEVICE)
    print(
        f"[Seed {seed}] USE24 post-EWC (MLP Ablation) -> "
        f"Acc: {metrics_use24_mlp['acc']:.4f} | "
        f"F1-Fake: {metrics_use24_mlp['f1_fake']:.4f}"
    )

    return metrics_use24_mlp


# ##############################################################################
# ### FASE 4: ENTRY POINT — PROTOCOLLO MULTI-SEED                          ###
# ##############################################################################

if __name__ == "__main__":

    print(f"Dispositivo di calcolo: {DEVICE}")

    (
        valid_graphs_pheme, labels_pheme,
        valid_graphs_use24, labels_use24,
    ) = load_and_build_graphs()

    # -- Loss condivisa da tutte le run, identica a run_hybrid.py --------------
    pos_w     = torch.tensor([0.5], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    res_use24_mlp = init_results_dict()

    for seed in SEEDS:
        print(f"\n{'=' * 50}")
        print(f"AVVIO RUN CON SEED: {seed}")
        print(f"{'=' * 50}")

        set_seed(seed)

        metrics_use24_mlp = run_ablation_for_seed(
            seed,
            valid_graphs_pheme, labels_pheme,
            valid_graphs_use24, labels_use24,
            criterion,
        )

        append_run_results(res_use24_mlp, metrics_use24_mlp)

    print("\n" + "*" * 60)
    print("STATISTICHE FINALI SU 5 SEED (MEDIA ± DEV. STD)")
    print("*" * 60)

    print_stats(
        "USE24 POST-EWC (MLP ABLATION — SENZA MESSAGE PASSING)",
        res_use24_mlp,
    )
