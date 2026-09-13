# =============================================================================
# FILE:    run_semantic.py
# SCOPO:   Script di esecuzione per la baseline semantica del progetto
#          "Fake News Detection: 2016 vs 2024". Esegue il protocollo
#          multi-seed completo: per ciascun seed, addestra
#          SemanticBaselineModel su PHEME (train/val/test 70/15/15),
#          valuta sul test set storico e sul campione a freddo di
#          USE24 (Concept Drift), quindi aggrega le metriche su tutti
#          i seed e stampa il report statistico finale (media ± std).
#          Le classi architetturali sono importate da models.semantic;
#          le utility di riproducibilita', metriche e aggregazione dei
#          risultati sono importate da utils.
# DIPENDENZE: torch, pandas, numpy, scikit-learn
# MODULO:  Esecuzione — Baseline Semantica
# =============================================================================


# ##############################################################################
# ### FASE 0: IMPORT E COSTANTI GLOBALI                                      ###
# ##############################################################################

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.semantic import TextEmbeddingDataset, SemanticBaselineModel
from utils import (
    DEVICE,
    set_seed,
    compute_classification_metrics,
    init_results_dict,
    append_run_results,
    print_stats,
)

# -- Percorsi Google Drive ----------------------------------------------------
DRIVE_BASE = "/content/drive/MyDrive/Tesi"
IN_PHEME   = f"{DRIVE_BASE}/Parquet_Finali/PHEME_Vectorized_FP16.parquet"
IN_USE24   = f"{DRIVE_BASE}/Parquet_Finali/USE24_Vectorized_FP16.parquet"

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
EPOCHS        = 10
BATCH_SIZE    = 32
LEARNING_RATE = 0.001


# ##############################################################################
# ### FASE 1: CARICAMENTO E MAPPING DEI DATASET (ESEGUITO UNA SOLA VOLTA)   ###
# ##############################################################################

def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carica e mappa i dataset PHEME e USE24 in memoria.

    Il caricamento e la mappatura delle label avvengono una sola volta,
    prima del ciclo multi-seed: gli split stocastici (train/val/test e
    il campionamento del 20% per USE24) sono invece ricalcolati per
    ciascun seed all'interno di ``train_and_evaluate_single_seed``,
    cosi' come nel notebook originale.

    Returns:
        Tupla ``(df_pheme, df_use24)`` con la colonna ``label_num``
        gia' calcolata e le righe non mappabili gia' filtrate.
    """
    print("Caricamento dataset PHEME e USE24 da Google Drive...")

    df_pheme = pd.read_parquet(IN_PHEME)
    target_col_pheme = (
        "status" if "status" in df_pheme.columns else "label"
    )
    df_pheme["label_num"] = (
        df_pheme[target_col_pheme]
        .astype(str).str.lower().str.strip()
        .map(PHEME_LABEL_MAP)
    )
    df_pheme = df_pheme.dropna(subset=["label_num"])
    df_pheme["label_num"] = df_pheme["label_num"].astype(int)

    df_use24 = pd.read_parquet(IN_USE24)
    target_col_use24 = (
        "status" if "status" in df_use24.columns else "label"
    )
    df_use24["label_num"] = (
        df_use24[target_col_use24]
        .astype(str).str.lower().str.strip()
        .map(USE24_LABEL_MAP)
    )
    df_use24 = df_use24.dropna(subset=["label_num"])
    df_use24["label_num"] = df_use24["label_num"].astype(int)

    print(
        f"Record PHEME: {len(df_pheme)} | "
        f"Record USE24: {len(df_use24)}"
    )
    return df_pheme, df_use24


# ##############################################################################
# ### FASE 2: TRAINING E VALUTAZIONE PER UN SINGOLO SEED                    ###
# ##############################################################################

def train_and_evaluate_single_seed(
    seed:     int,
    df_pheme: pd.DataFrame,
    df_use24: pd.DataFrame,
) -> tuple[dict, dict, list[float], list[float]]:
    """Esegue il ciclo completo di training e valutazione per un singolo seed.

    Riproduce esattamente la logica del notebook originale per una
    singola run del protocollo multi-seed:
      1. Split stratificato di PHEME in train/val/test (70/15/15).
      2. Campionamento stratificato del 20% di USE24 (test a freddo).
      3. Inizializzazione di un nuovo SemanticBaselineModel.
      4. Training su PHEME con validazione ad ogni epoca.
      5. Valutazione sul test set di PHEME.
      6. Valutazione sul campione USE24 (Concept Drift).

    Tutti gli split usano ``random_state=seed``, garantendo che la
    suddivisione dei dati sia deterministica e riproducibile per ogni
    singolo seed del protocollo.

    Args:
        seed:     Seed della run corrente. Deve essere gia' stato
            propagato ai generatori casuali tramite ``set_seed(seed)``
            prima della chiamata.
        df_pheme: DataFrame PHEME con colonna ``label_num`` gia'
            calcolata da ``load_datasets``.
        df_use24: DataFrame USE24 con colonna ``label_num`` gia'
            calcolata da ``load_datasets``.

    Returns:
        Tupla ``(metrics_pheme, metrics_use24, train_loss_history,
        val_loss_history)`` dove i primi due elementi sono i dizionari
        prodotti da ``compute_classification_metrics`` e gli ultimi
        due sono le liste di loss medie per epoca (una voce per
        epoca di training).
    """
    # -- 1. Split PHEME (70% train, 15% val, 15% test) -----------------------
    df_train, df_temp = train_test_split(
        df_pheme, test_size=0.30, random_state=seed,
        stratify=df_pheme["label_num"],
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.50, random_state=seed,
        stratify=df_temp["label_num"],
    )

    train_loader = DataLoader(
        TextEmbeddingDataset(df_train),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TextEmbeddingDataset(df_val),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    test_loader_pheme = DataLoader(
        TextEmbeddingDataset(df_test),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # -- 2. Split USE24 (campione 20% stratificato) ---------------------------
    df_use24_sample, _ = train_test_split(
        df_use24, train_size=0.20, random_state=seed,
        stratify=df_use24["label_num"],
    )
    test_loader_use24 = DataLoader(
        TextEmbeddingDataset(df_use24_sample),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # -- 3. Inizializzazione modello, loss, ottimizzatore ---------------------
    model     = SemanticBaselineModel().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss_history: list[float] = []
    val_loss_history:   list[float] = []

    # -- 4. Training loop con validazione per epoca ---------------------------
    for _ in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            predizioni = model(batch_x).squeeze(1)
            loss = criterion(predizioni, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                predizioni = model(batch_x).squeeze(1)
                loss = criterion(predizioni, batch_y)
                val_loss += loss.item()

        train_loss_history.append(train_loss / len(train_loader))
        val_loss_history.append(val_loss / len(val_loader))

    # -- 5. Funzione di valutazione (interna: dipende dal modello locale) ----
    def evaluate_model(loader: DataLoader) -> tuple[list[float], list[float]]:
        """Esegue l'inferenza su un DataLoader e raccoglie predizioni e label.

        Args:
            loader: DataLoader del set da valutare.

        Returns:
            Tupla ``(all_labels, all_preds)`` con le liste di label
            vere e predizioni binarie (0/1) accumulate su tutti i
            batch.
        """
        model.eval()
        all_preds:  list[float] = []
        all_labels: list[float] = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(DEVICE)
                probs = torch.sigmoid(model(batch_x).squeeze(1))
                all_preds.extend((probs > 0.5).float().cpu().numpy())
                all_labels.extend(batch_y.numpy())
        return all_labels, all_preds

    # -- 6. Valutazione su PHEME (test set storico) ---------------------------
    labels_pheme, preds_pheme = evaluate_model(test_loader_pheme)
    metrics_pheme = compute_classification_metrics(
        labels_pheme, preds_pheme
    )
    print(
        f"PHEME  -> Acc: {metrics_pheme['acc']:.4f} | "
        f"P-Fake: {metrics_pheme['p_fake']:.4f} | "
        f"R-Fake: {metrics_pheme['r_fake']:.4f} | "
        f"F1-Fake: {metrics_pheme['f1_fake']:.4f}"
    )

    # -- 7. Valutazione su USE24 (Concept Drift a freddo) ---------------------
    labels_use24, preds_use24 = evaluate_model(test_loader_use24)
    metrics_use24 = compute_classification_metrics(
        labels_use24, preds_use24
    )
    print(
        f"USE24  -> Acc: {metrics_use24['acc']:.4f} | "
        f"P-Fake: {metrics_use24['p_fake']:.4f} | "
        f"R-Fake: {metrics_use24['r_fake']:.4f} | "
        f"F1-Fake: {metrics_use24['f1_fake']:.4f}"
    )

    return metrics_pheme, metrics_use24, train_loss_history, val_loss_history


# ##############################################################################
# ### FASE 3: ENTRY POINT — PROTOCOLLO MULTI-SEED                          ###
# ##############################################################################

if __name__ == "__main__":

    df_pheme, df_use24 = load_datasets()

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
        ) = train_and_evaluate_single_seed(seed, df_pheme, df_use24)

        append_run_results(results_pheme, metrics_pheme)
        results_pheme["train_loss_history"].append(train_loss_history)
        results_pheme["val_loss_history"].append(val_loss_history)

        append_run_results(results_use24, metrics_use24)

    print("\n" + "*" * 60)
    print("STATISTICHE FINALI SU 5 SEED (MEDIA ± DEV. STD)")
    print("*" * 60)

    print_stats("PHEME (TEST SET STORICO)", results_pheme)
    print_stats("USE24 (CONCEPT DRIFT A FREDDO)", results_use24)
