# =============================================================================
# FILE:    run_topological.py
# SCOPO:   Script di esecuzione per la baseline topologica del progetto
#          "Fake News Detection: 2016 vs 2024". Esegue il protocollo
#          multi-seed completo: per ciascun seed, addestra
#          BiGCNBaselineModel sulle cascate di propagazione di PHEME
#          (train/val/test 70/15/15), valuta sul test set storico e sul
#          campione a freddo di USE24 (Concept Drift), quindi aggrega le
#          metriche su tutti i seed e stampa il report statistico finale
#          (media ± deviazione standard).
#          L'architettura Bi-GCN e' importata da models.topological; le
#          utility di riproducibilita', metriche e aggregazione dei
#          risultati sono importate da utils.
# DIPENDENZE: torch, torch_geometric, pandas, numpy, scikit-learn
# MODULO:  Esecuzione — Baseline Topologica (Bi-GCN)
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

from models.topological import BiGCNBaselineModel
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
EPOCHS        = 15
BATCH_SIZE    = 32
LEARNING_RATE = 0.0005


# ##############################################################################
# ### FASE 1: CARICAMENTO E COSTRUZIONE DEI GRAFI (ESEGUITO UNA SOLA VOLTA) ###
# ##############################################################################

def load_and_build_graphs() -> tuple[list, list[int], list, list[int]]:
    """Carica i grafi PyG di PHEME e USE24 e inietta le label numeriche.

    Per ciascun dataset, esegue il join tra la lista di grafi salvata
    in formato .pt (con label sentinella) e le annotazioni testuali del
    Parquet vettorizzato, usando ``root_id`` (lato grafi) e ``node_id``
    (lato Parquet, o ``tweet_id`` come fallback) come chiave.

    Questo caricamento avviene una sola volta, prima del ciclo
    multi-seed: gli split stocastici (train/val/test per PHEME e il
    campionamento del 20% per USE24) vengono invece ricalcolati per
    ciascun seed all'interno di ``train_and_evaluate_single_seed``,
    esattamente come nel notebook originale.

    Returns:
        Tupla ``(valid_graphs_pheme, labels_pheme, valid_graphs_use24,
        labels_use24)`` dove le liste di grafi contengono oggetti
        ``torch_geometric.data.Data`` con ``g.y`` gia' impostato a
        ``torch.tensor([label], dtype=torch.float32)``, e le liste di
        label sono le corrispondenti liste parallele di interi 0/1
        usate per lo stratify negli split scikit-learn.
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

    pheme_graphs_raw = torch.load(
        IN_PHEME_GRAPHS, weights_only=False
    )
    valid_graphs_pheme: list     = []
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

    use24_graphs_raw = torch.load(
        IN_USE24_GRAPHS, weights_only=False
    )
    valid_graphs_use24: list     = []
    labels_use24:       list[int] = []

    for g in use24_graphs_raw:
        root_id = str(g.root_id)
        if root_id in label_dict_use24:
            vera_label = label_dict_use24[root_id]
            g.y = torch.tensor([vera_label], dtype=torch.float32)
            valid_graphs_use24.append(g)
            labels_use24.append(vera_label)

    print(
        f"Grafi accoppiati validi -> "
        f"PHEME: {len(valid_graphs_pheme)} | "
        f"USE24: {len(valid_graphs_use24)}"
    )

    return (
        valid_graphs_pheme, labels_pheme,
        valid_graphs_use24, labels_use24,
    )


# ##############################################################################
# ### FASE 2: TRAINING E VALUTAZIONE PER UN SINGOLO SEED                    ###
# ##############################################################################

def train_and_evaluate_single_seed(
    seed:                int,
    valid_graphs_pheme:  list,
    labels_pheme:        list[int],
    valid_graphs_use24:  list,
    labels_use24:        list[int],
) -> tuple[dict, dict, list[float], list[float]]:
    """Esegue il ciclo completo di training e valutazione per un singolo seed.

    Riproduce esattamente la logica del notebook originale per una
    singola run del protocollo multi-seed:
      1. Split stratificato dei grafi PHEME in train/val/test (70/15/15).
      2. Campionamento stratificato del 20% dei grafi USE24 (test a
         freddo).
      3. Inizializzazione di un nuovo BiGCNBaselineModel.
      4. Training su PHEME con validazione ad ogni epoca, gestendo i
         batch PyG (``batch_data.x``, ``batch_data.edge_index``,
         ``batch_data.batch``, ``batch_data.y``).
      5. Valutazione sul test set di PHEME.
      6. Valutazione sul campione USE24 (Concept Drift).

    Tutti gli split usano ``random_state=seed``, garantendo che la
    suddivisione dei grafi sia deterministica e riproducibile per ogni
    singolo seed del protocollo.

    Args:
        seed: Seed della run corrente. Deve essere gia' stato
            propagato ai generatori casuali tramite ``set_seed(seed)``
            prima della chiamata.
        valid_graphs_pheme: Lista di grafi PHEME con label gia'
            iniettata, prodotta da ``load_and_build_graphs``.
        labels_pheme: Lista parallela di interi 0/1 per lo stratify
            degli split di PHEME.
        valid_graphs_use24: Lista di grafi USE24 con label gia'
            iniettata, prodotta da ``load_and_build_graphs``.
        labels_use24: Lista parallela di interi 0/1 per lo stratify
            del campionamento di USE24.

    Returns:
        Tupla ``(metrics_pheme, metrics_use24, train_loss_history,
        val_loss_history)`` dove i primi due elementi sono i dizionari
        prodotti da ``compute_classification_metrics`` e gli ultimi due
        sono le liste di loss medie per epoca (una voce per epoca di
        training).
    """
    # -- 1. Split PHEME (70% train, 15% val, 15% test) -----------------------
    train_graphs, temp_graphs, _, temp_labels = train_test_split(
        valid_graphs_pheme, labels_pheme,
        test_size=0.30, random_state=seed, stratify=labels_pheme,
    )
    val_graphs, test_graphs = train_test_split(
        temp_graphs,
        test_size=0.50, random_state=seed, stratify=temp_labels,
    )

    train_loader = DataLoader(
        train_graphs, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_graphs, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader_pheme = DataLoader(
        test_graphs, batch_size=BATCH_SIZE, shuffle=False
    )

    # -- 2. Split USE24 (campione 20% stratificato, test a freddo) -----------
    use24_sample_graphs, _ = train_test_split(
        valid_graphs_use24, train_size=0.20,
        random_state=seed, stratify=labels_use24,
    )
    test_loader_use24 = DataLoader(
        use24_sample_graphs, batch_size=BATCH_SIZE, shuffle=False
    )

    # -- 3. Inizializzazione modello, loss, ottimizzatore ---------------------
    model = BiGCNBaselineModel(
        in_feats=768, hidden_feats=64
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss_history: list[float] = []
    val_loss_history:   list[float] = []

    # -- 4. Training loop con validazione per epoca ---------------------------
    for _ in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_data in train_loader:
            batch_data = batch_data.to(DEVICE)
            optimizer.zero_grad()

            out    = model(batch_data).view(-1)
            y_true = batch_data.y.view(-1)

            loss = criterion(out, y_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(DEVICE)
                out    = model(batch_data).view(-1)
                y_true = batch_data.y.view(-1)
                loss = criterion(out, y_true)
                val_loss += loss.item()

        train_loss_history.append(train_loss / len(train_loader))
        val_loss_history.append(val_loss / len(val_loader))

    # -- 5. Funzione di valutazione (interna: dipende dal modello locale) ----
    def evaluate_model(
        loader: DataLoader,
    ) -> tuple[list[float], list[float]]:
        """Esegue l'inferenza su un DataLoader PyG e raccoglie predizioni e label.

        Args:
            loader: DataLoader PyG del set da valutare.

        Returns:
            Tupla ``(all_labels, all_preds)`` con le liste di label
            vere e predizioni binarie (0/1) accumulate su tutti i
            batch.
        """
        model.eval()
        all_preds:  list[float] = []
        all_labels: list[float] = []
        with torch.no_grad():
            for batch_data in loader:
                batch_data = batch_data.to(DEVICE)
                logits = model(batch_data).view(-1)
                probs  = torch.sigmoid(logits)

                all_preds.extend((probs > 0.5).float().cpu().numpy())
                all_labels.extend(
                    batch_data.y.view(-1).cpu().numpy()
                )
        return all_labels, all_preds

    # -- 6. Valutazione su PHEME (test set storico) ---------------------------
    labels_p, preds_p = evaluate_model(test_loader_pheme)
    metrics_pheme = compute_classification_metrics(labels_p, preds_p)
    print(
        f"PHEME  -> Acc: {metrics_pheme['acc']:.4f} | "
        f"P-Fake: {metrics_pheme['p_fake']:.4f} | "
        f"R-Fake: {metrics_pheme['r_fake']:.4f} | "
        f"F1-Fake: {metrics_pheme['f1_fake']:.4f}"
    )

    # -- 7. Valutazione su USE24 (Concept Drift a freddo) ---------------------
    labels_u, preds_u = evaluate_model(test_loader_use24)
    metrics_use24 = compute_classification_metrics(labels_u, preds_u)
    print(
        f"USE24  -> Acc: {metrics_use24['acc']:.4f} | "
        f"P-Fake: {metrics_use24['p_fake']:.4f} | "
        f"R-Fake: {metrics_use24['r_fake']:.4f} | "
        f"F1-Fake: {metrics_use24['f1_fake']:.4f}"
    )

    return (
        metrics_pheme, metrics_use24,
        train_loss_history, val_loss_history,
    )


# ##############################################################################
# ### FASE 3: ENTRY POINT — PROTOCOLLO MULTI-SEED                          ###
# ##############################################################################

if __name__ == "__main__":

    (
        valid_graphs_pheme, labels_pheme,
        valid_graphs_use24, labels_use24,
    ) = load_and_build_graphs()

    results_pheme = init_results_dict(include_loss_history=True)
    results_use24 = init_results_dict(include_loss_history=False)

    for seed in SEEDS:
        print(f"\n{'=' * 50}")
        print(f"AVVIO RUN CON SEED: {seed}")
        print(f"{'=' * 50}")

        set_seed(seed)

        (
            metrics_pheme,
            metrics_use24,
            train_loss_history,
            val_loss_history,
        ) = train_and_evaluate_single_seed(
            seed,
            valid_graphs_pheme, labels_pheme,
            valid_graphs_use24, labels_use24,
        )

        append_run_results(results_pheme, metrics_pheme)
        results_pheme["train_loss_history"].append(train_loss_history)
        results_pheme["val_loss_history"].append(val_loss_history)

        append_run_results(results_use24, metrics_use24)

    print("\n" + "*" * 60)
    print("STATISTICHE FINALI SU 5 SEED (MEDIA ± DEV. STD)")
    print("*" * 60)

    print_stats("PHEME (TEST SET STORICO)", results_pheme)
    print_stats("USE24 (CONCEPT DRIFT A FREDDO)", results_use24)
