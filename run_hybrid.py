# =============================================================================
# FILE:    run_hybrid.py
# SCOPO:   Script di esecuzione per il modello ibrido Gated Bi-GCN con
#          Continual Learning tramite Elastic Weight Consolidation (EWC)
#          del progetto "Fake News Detection: 2016 vs 2024". Per
#          ciascun seed del protocollo multi-run esegue in sequenza:
#            (A) Split stratificato di PHEME e USE24.
#            (B) Training storico su PHEME (15 epoche), valutazione su
#                PHEME e sul campione a freddo di USE24 (Concept Drift),
#                estrazione del comportamento del gate (alpha).
#            (C) Estrazione della Fisher Information Matrix su PHEME e
#                fine-tuning Naive (senza EWC) su USE24, per misurare la
#                Dimenticanza Catastrofica sul test set di PHEME.
#            (D) Fine-tuning EWC su USE24 (lambda=50000), valutazione su
#                PHEME (Backward Transfer) e USE24 (adattamento),
#                estrazione del gate post-EWC.
#            (E) Analisi di sensitivita' su lambda: per ciascun valore
#                in [0, 10000, 25000, 50000, 100000], fine-tuning breve
#                (3 epoche) e misurazione della F1-Fake di retention su
#                PHEME.
#          L'Ablation Study con HybridGatedMLP (Fase 5 del notebook) e'
#          deliberatamente esclusa da questo script: verra' gestita in
#          un modulo separato.
#          L'architettura e' importata da models.hybrid, la logica EWC
#          da models.ewc, le utility di riproducibilita', metriche e
#          aggregazione dei risultati da utils.
# DIPENDENZE: torch, torch_geometric, pandas, numpy, scikit-learn
# MODULO:  Esecuzione — Modello Ibrido + Continual Learning (EWC)
# =============================================================================


# ##############################################################################
# ### FASE 0: IMPORT E COSTANTI GLOBALI                                      ###
# ##############################################################################

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# NOTA: DataLoader di PyTorch Geometric, non quello standard di
# torch.utils.data — necessario per il collate corretto dei batch di
# grafi (super-grafo sparso a blocchi diagonali).
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from models.hybrid import HybridGatedBiGCN
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
CLIP_NORM      = 1.0

EPOCHS_MAIN    = 15   # Training storico su PHEME (Fase B)
EPOCHS_FT      = 5    # Fine-tuning Naive ed EWC su USE24 (Fasi C, D)
EPOCHS_LAMBDA  = 3    # Fine-tuning breve per la sensitivity analysis (Fase E)

# -- Coefficiente EWC per il fine-tuning principale (Fase D) -----------------
LAMBDA_EWC     = 50000

# -- Valori di lambda per l'analisi di sensitivita' (Fase E) -----------------
LAMBDA_VALS: list[int] = [0, 10000, 25000, 50000, 100000]


# ##############################################################################
# ### FASE 1: CARICAMENTO E COSTRUZIONE DEI GRAFI (ESEGUITO UNA SOLA VOLTA) ###
# ##############################################################################

def load_and_build_graphs() -> tuple[list, list[int], list, list[int]]:
    """Carica i grafi PyG di PHEME e USE24 e inietta le label numeriche.

    Per ciascun dataset, esegue il join tra la lista di grafi salvata in
    formato .pt (con label sentinella) e le annotazioni testuali del
    Parquet vettorizzato, usando ``root_id`` (lato grafi) e ``node_id``
    (lato Parquet) come chiave.

    Identica alla funzione omonima in ``run_topological.py``: la logica
    di caricamento e mapping delle label e' condivisa tra la baseline
    topologica e il modello ibrido, poiche' entrambi operano sugli
    stessi grafi di propagazione.

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
# ### FASE 2: UTILITY DI VALUTAZIONE E ISPEZIONE DEL GATE                   ###
# ##############################################################################

def evaluate_model(
    model:  HybridGatedBiGCN,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Esegue l'inferenza su un DataLoader PyG e calcola le metriche complete.

    Args:
        model:  Istanza di ``HybridGatedBiGCN`` da valutare.
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


def get_alpha(
    model:  HybridGatedBiGCN,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Calcola il valore medio del gate adattivo su un intero DataLoader.

    Esegue un forward pass su ogni batch per popolare
    ``model.last_alpha`` (impostato internamente da
    ``HybridGatedBiGCN.forward``), quindi ne calcola la media per
    batch e infine la media complessiva su tutti i batch. Un valore
    di alpha vicino a 1 indica che il gate si affida prevalentemente
    al ramo semantico; un valore vicino a 0 indica una prevalenza del
    ramo topologico.

    Args:
        model:  Istanza di ``HybridGatedBiGCN`` gia' addestrata.
        loader: DataLoader PyG sul quale ispezionare il comportamento
            del gate.
        device: Dispositivo di calcolo.

    Returns:
        Media (float) del valore di alpha su tutti i batch del loader.
    """
    model.eval()
    alphas: list[float] = []
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            _ = model(batch_data)
            alphas.append(model.last_alpha.mean().item())
    return float(np.mean(alphas))


# ##############################################################################
# ### FASE 3: ESPERIMENTO COMPLETO PER UN SINGOLO SEED                      ###
# ##############################################################################

def run_experiment_for_seed(
    seed:                int,
    valid_graphs_pheme:  list,
    labels_pheme:        list[int],
    valid_graphs_use24:  list,
    labels_use24:        list[int],
    criterion:           nn.Module,
) -> dict:
    """Esegue l'intero protocollo Ibrido + EWC per un singolo seed.

    Riproduce esattamente la sequenza di fasi del notebook originale:

    (A) Split stratificato di PHEME (70/15/15) e USE24 (80% train
        fine-tuning / 20% test, usato anche come cold test).
    (B) Training storico: un nuovo ``HybridGatedBiGCN`` viene
        addestrato da zero su PHEME per ``EPOCHS_MAIN`` epoche, con
        validazione ad ogni epoca. Il modello risultante (``model``)
        e' valutato sul test set di PHEME e, senza alcun
        fine-tuning, sul campione USE24 (test a freddo / Concept
        Drift). Il comportamento del gate viene ispezionato su
        entrambi i test set.
    (C) Naive Fine-Tuning: la Fisher Information Matrix viene
        calcolata sul training set di PHEME a partire dal modello
        storico. Un clone indipendente (``model_naive``, tramite
        ``copy.deepcopy``) viene poi addestrato su USE24 senza alcun
        vincolo EWC, per ``EPOCHS_FT`` epoche, e infine valutato sul
        test set di PHEME per quantificare la Dimenticanza
        Catastrofica.
    (D) EWC Fine-Tuning: un secondo clone indipendente
        (``model_ewc``) viene addestrato su USE24 con la penale EWC
        (coefficiente ``LAMBDA_EWC``) calcolata da
        ``compute_ewc_penalty`` rispetto alla FIM e ai pesi storici
        estratti in (C). Il modello risultante e' valutato sia su
        PHEME (Backward Transfer) che su USE24 (adattamento al nuovo
        task), e il suo gate viene ispezionato su USE24.
    (E) Lambda Sensitivity: per ciascun valore in ``LAMBDA_VALS``, un
        ulteriore clone del modello storico viene sottoposto a un
        fine-tuning breve (``EPOCHS_LAMBDA`` epoche) su USE24 con
        quel coefficiente EWC; viene registrata la sola F1-Fake sul
        test set di PHEME, come misura di retention della memoria
        storica in funzione di lambda.

    Ogni clone (``model_naive``, ``model_ewc``, i modelli della fase
    di sensitivity) e' generato tramite ``copy.deepcopy(model)``
    a partire dal modello storico (B), mai l'uno dall'altro: questo
    garantisce che ciascun ramo sperimentale parta esattamente dagli
    stessi pesi ``theta*_A`` e che nessun fine-tuning influenzi gli
    altri rami.

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
            degli split di USE24.
        criterion: Funzione di loss condivisa da tutte le fasi
            (``BCEWithLogitsLoss`` con ``pos_weight``), inizializzata
            una sola volta fuori dal ciclo sui seed.

    Returns:
        Dizionario con le chiavi:
          ``metrics_pheme_base``, ``metrics_use24_cold``,
          ``train_loss_history``, ``val_loss_history``,
          ``alpha_pheme``, ``alpha_use24_cold``,
          ``metrics_pheme_naive``,
          ``metrics_pheme_ewc``, ``metrics_use24_ewc``,
          ``alpha_use24_ewc``,
          ``lambda_f1``: dizionario ``{lambda_val: f1_fake}`` per
            questo seed.
    """
    # -- (A) Split PHEME (70% train, 15% val, 15% test) -----------------------
    train_g, temp_g, _, temp_l = train_test_split(
        valid_graphs_pheme, labels_pheme,
        test_size=0.30, random_state=seed, stratify=labels_pheme,
    )
    val_g, test_g = train_test_split(
        temp_g, test_size=0.50, random_state=seed, stratify=temp_l,
    )

    train_loader_ph = DataLoader(
        train_g, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader_ph = DataLoader(
        val_g, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader_ph = DataLoader(
        test_g, batch_size=BATCH_SIZE, shuffle=False
    )

    # -- (A) Split USE24 (80% train fine-tuning, 20% test/cold) ---------------
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
    # (B) FASE 1 — Training storico su PHEME                            #
    # ------------------------------------------------------------------ #
    model = HybridGatedBiGCN().to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_loss_history: list[float] = []
    val_loss_history:   list[float] = []

    for _ in range(EPOCHS_MAIN):
        model.train()
        train_loss = 0.0
        for batch_data in train_loader_ph:
            batch_data = batch_data.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(
                model(batch_data).view(-1), batch_data.y.view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CLIP_NORM
            )
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader_ph:
                batch_data = batch_data.to(DEVICE)
                val_loss += criterion(
                    model(batch_data).view(-1), batch_data.y.view(-1)
                ).item()

        train_loss_history.append(train_loss / len(train_loader_ph))
        val_loss_history.append(val_loss / len(val_loader_ph))

    metrics_pheme_base = evaluate_model(model, test_loader_ph, DEVICE)
    metrics_use24_cold = evaluate_model(model, test_loader_u24, DEVICE)

    alpha_pheme      = get_alpha(model, test_loader_ph, DEVICE)
    alpha_use24_cold = get_alpha(model, test_loader_u24, DEVICE)

    print(
        f"[Seed {seed}] PHEME base -> "
        f"Acc: {metrics_pheme_base['acc']:.4f} | "
        f"F1-Fake: {metrics_pheme_base['f1_fake']:.4f}"
    )
    print(
        f"[Seed {seed}] USE24 cold -> "
        f"Acc: {metrics_use24_cold['acc']:.4f} | "
        f"F1-Fake: {metrics_use24_cold['f1_fake']:.4f}"
    )

    # ------------------------------------------------------------------ #
    # (C) FASE 2 — Fisher Matrix + Naive Fine-Tuning (Dimenticanza)      #
    # ------------------------------------------------------------------ #
    fisher_dict, opt_params = compute_fisher_matrix(
        model=model, loader=train_loader_ph,
        criterion=criterion, device=DEVICE,
    )

    model_naive = copy.deepcopy(model)
    opt_naive = optim.AdamW(
        model_naive.parameters(),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    model_naive.train()
    for _ in range(EPOCHS_FT):
        for batch_data in train_loader_u24:
            batch_data = batch_data.to(DEVICE)
            opt_naive.zero_grad()
            loss = criterion(
                model_naive(batch_data).view(-1), batch_data.y.view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model_naive.parameters(), CLIP_NORM
            )
            opt_naive.step()

    metrics_pheme_naive = evaluate_model(
        model_naive, test_loader_ph, DEVICE
    )
    print(
        f"[Seed {seed}] PHEME Naive (Backward Transfer) -> "
        f"F1-Fake: {metrics_pheme_naive['f1_fake']:.4f}"
    )

    # ------------------------------------------------------------------ #
    # (D) FASE 3 — EWC Fine-Tuning (Resilienza)                          #
    # ------------------------------------------------------------------ #
    model_ewc = copy.deepcopy(model)
    opt_ewc = optim.AdamW(
        model_ewc.parameters(),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    model_ewc.train()
    for _ in range(EPOCHS_FT):
        for batch_data in train_loader_u24:
            batch_data = batch_data.to(DEVICE)
            opt_ewc.zero_grad()

            l_task = criterion(
                model_ewc(batch_data).view(-1), batch_data.y.view(-1)
            )
            # compute_ewc_penalty restituisce gia' il termine scalato
            # per lambda/2: non applicare ulteriori fattori di scala.
            l_ewc = compute_ewc_penalty(
                model=model_ewc,
                fisher_dict=fisher_dict,
                opt_params=opt_params,
                lambda_val=LAMBDA_EWC,
            )
            loss = l_task + l_ewc

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model_ewc.parameters(), CLIP_NORM
            )
            opt_ewc.step()

    metrics_pheme_ewc = evaluate_model(model_ewc, test_loader_ph, DEVICE)
    metrics_use24_ewc = evaluate_model(model_ewc, test_loader_u24, DEVICE)
    alpha_use24_ewc   = get_alpha(model_ewc, test_loader_u24, DEVICE)

    print(
        f"[Seed {seed}] PHEME EWC (Backward Transfer) -> "
        f"F1-Fake: {metrics_pheme_ewc['f1_fake']:.4f}"
    )
    print(
        f"[Seed {seed}] USE24 post-EWC -> "
        f"F1-Fake: {metrics_use24_ewc['f1_fake']:.4f}"
    )

    # ------------------------------------------------------------------ #
    # (E) FASE 4 — Lambda Sensitivity Analysis                           #
    # ------------------------------------------------------------------ #
    # NOTA DI FEDELTA': nel notebook originale questo loop non applica
    # gradient clipping (a differenza delle Fasi B, C, D). Il
    # comportamento e' preservato esattamente qui.
    lambda_f1: dict[int, float] = {}

    for lambda_val in LAMBDA_VALS:
        model_sens = copy.deepcopy(model)
        opt_sens = optim.AdamW(
            model_sens.parameters(),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
        model_sens.train()
        for _ in range(EPOCHS_LAMBDA):
            for batch_data in train_loader_u24:
                batch_data = batch_data.to(DEVICE)
                opt_sens.zero_grad()

                l_task = criterion(
                    model_sens(batch_data).view(-1),
                    batch_data.y.view(-1),
                )
                l_ewc = compute_ewc_penalty(
                    model=model_sens,
                    fisher_dict=fisher_dict,
                    opt_params=opt_params,
                    lambda_val=lambda_val,
                )
                loss = l_task + l_ewc

                loss.backward()
                opt_sens.step()

        metrics_sens = evaluate_model(model_sens, test_loader_ph, DEVICE)
        lambda_f1[lambda_val] = metrics_sens["f1_fake"]

    return {
        "metrics_pheme_base":  metrics_pheme_base,
        "metrics_use24_cold":  metrics_use24_cold,
        "train_loss_history":  train_loss_history,
        "val_loss_history":    val_loss_history,
        "alpha_pheme":         alpha_pheme,
        "alpha_use24_cold":    alpha_use24_cold,
        "metrics_pheme_naive": metrics_pheme_naive,
        "metrics_pheme_ewc":   metrics_pheme_ewc,
        "metrics_use24_ewc":   metrics_use24_ewc,
        "alpha_use24_ewc":     alpha_use24_ewc,
        "lambda_f1":           lambda_f1,
    }


# ##############################################################################
# ### FASE 4: ENTRY POINT — PROTOCOLLO MULTI-SEED                          ###
# ##############################################################################

if __name__ == "__main__":

    print(f"Dispositivo di calcolo: {DEVICE}")

    (
        valid_graphs_pheme, labels_pheme,
        valid_graphs_use24, labels_use24,
    ) = load_and_build_graphs()

    # -- Loss condivisa da tutte le fasi e da tutti i seed --------------------
    pos_w     = torch.tensor([0.5], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # -- Dizionari di aggregazione multi-seed ----------------------------------
    res_pheme_base  = init_results_dict(include_loss_history=True)
    res_use24_cold  = init_results_dict()
    res_pheme_naive = init_results_dict()
    res_pheme_ewc   = init_results_dict()
    res_use24_ewc   = init_results_dict()

    alpha_vals: dict[str, list[float]] = {
        "pheme":      [],
        "use24_cold": [],
        "use24_ewc":  [],
    }
    res_lambda: dict[int, list[float]] = {l: [] for l in LAMBDA_VALS}

    for seed in SEEDS:
        print(f"\n{'=' * 50}")
        print(f"AVVIO RUN CON SEED: {seed}")
        print(f"{'=' * 50}")

        set_seed(seed)

        results = run_experiment_for_seed(
            seed,
            valid_graphs_pheme, labels_pheme,
            valid_graphs_use24, labels_use24,
            criterion,
        )

        append_run_results(res_pheme_base, results["metrics_pheme_base"])
        res_pheme_base["train_loss_history"].append(
            results["train_loss_history"]
        )
        res_pheme_base["val_loss_history"].append(
            results["val_loss_history"]
        )

        append_run_results(res_use24_cold, results["metrics_use24_cold"])
        append_run_results(res_pheme_naive, results["metrics_pheme_naive"])
        append_run_results(res_pheme_ewc, results["metrics_pheme_ewc"])
        append_run_results(res_use24_ewc, results["metrics_use24_ewc"])

        alpha_vals["pheme"].append(results["alpha_pheme"])
        alpha_vals["use24_cold"].append(results["alpha_use24_cold"])
        alpha_vals["use24_ewc"].append(results["alpha_use24_ewc"])

        for lambda_val in LAMBDA_VALS:
            res_lambda[lambda_val].append(
                results["lambda_f1"][lambda_val]
            )

    # -- Report statistico finale -----------------------------------------------
    print("\n" + "*" * 60)
    print("STATISTICHE FINALI SU 5 SEED (MEDIA ± DEV. STD)")
    print("*" * 60)

    print_stats("1. PHEME BASELINE", res_pheme_base)
    print_stats("2. USE24 (COLD TEST)", res_use24_cold)
    print_stats("3. PHEME BACKWARD TRANSFER (NAIVE)", res_pheme_naive)
    print_stats("4. PHEME BACKWARD TRANSFER (EWC)", res_pheme_ewc)
    print_stats("5. USE24 POST-EWC (BiGCN)", res_use24_ewc)

    print("\n--- 6. GATE ALPHA BEHAVIOR ---")
    print(
        f"Alpha PHEME storici : "
        f"{np.mean(alpha_vals['pheme']):.4f} ± "
        f"{np.std(alpha_vals['pheme']):.4f}"
    )
    print(
        f"Alpha USE24 freddo  : "
        f"{np.mean(alpha_vals['use24_cold']):.4f} ± "
        f"{np.std(alpha_vals['use24_cold']):.4f}"
    )
    print(
        f"Alpha USE24 post-EWC: "
        f"{np.mean(alpha_vals['use24_ewc']):.4f} ± "
        f"{np.std(alpha_vals['use24_ewc']):.4f}"
    )

    print("\n--- 7. LAMBDA SENSITIVITY ---")
    for lambda_val in LAMBDA_VALS:
        values = res_lambda[lambda_val]
        print(
            f"Lambda {lambda_val:6d} -> "
            f"F1-Fake Ritenzione: "
            f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        )
