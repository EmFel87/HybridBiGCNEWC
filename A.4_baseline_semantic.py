# =============================================================================
# FILE:    baseline_semantic.py
# SCOPO:   Baseline semantica per il rilevamento delle Fake News tramite
#          embedding testuali BERTweet (vinai/bertweet-base).
#          Il modulo esegue tre esperimenti in sequenza:
#            1. Ablation Study su LIAR (benchmark di generalizzazione).
#            2. Training + Validazione + Test su PHEME (dati storici 2016).
#            3. Test a freddo su USE24 (misurazione del Concept Drift 2024).
#          I risultati del test a freddo costituiscono la baseline di
#          riferimento per la valutazione del modello ibrido Bi-GCN + EWC.
# DIPENDENZE: torch, scikit-learn, pandas, numpy
# MODULO:  Baseline Semantica — esperimento indipendente dai moduli ETL
# =============================================================================


# ##############################################################################
# ### FASE 0: IMPORT E COSTANTI GLOBALI                                      ###
# ##############################################################################

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from typing import Optional

# -- Dispositivo di calcolo ---------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Percorsi Google Drive (modificare in base alla struttura del progetto) ---
DRIVE_BASE = "/content/drive/MyDrive/Tesi"
IN_LIAR    = f"{DRIVE_BASE}/Parquet_Finali/LIAR_Vectorized_FP16.parquet"
IN_PHEME   = f"{DRIVE_BASE}/Parquet_Finali/PHEME_Vectorized_FP16.parquet"
IN_USE24   = f"{DRIVE_BASE}/Parquet_Finali/USE24_Vectorized_FP16.parquet"

# -- Iperparametri del modello ------------------------------------------------
INPUT_DIM     = 768    # Dimensione embedding [CLS] di BERTweet
HIDDEN_DIM    = 256    # Dimensione del layer nascosto intermedio
DROPOUT_RATE  = 0.3    # Probabilita' di dropout durante il training
LEARNING_RATE = 1e-3   # Learning rate per Adam
BATCH_SIZE    = 32
NUM_EPOCHS    = 10

# -- Mapping label testuale -> classe binaria (0 = Fake, 1 = Real) -----------
#    Ogni dataset usa un vocabolario di etichette diverso; i dizionari
#    centralizzati qui sotto garantiscono coerenza e unicita' della logica
#    di conversione lungo tutto il modulo.

# LIAR: scala a sei livelli di verita' PolitiFact -> binario
LIAR_LABEL_MAP: dict[str, int] = {
    "true":        1,
    "mostly-true": 1,
    "half-true":   1,
    "barely-true": 0,
    "false":       0,
    "pants-fire":  0,
}

# PHEME: annotazioni binarie originali (varianti ortografiche BR/US)
PHEME_LABEL_MAP: dict[str, int] = {
    "rumour":     0,
    "rumor":      0,
    "non-rumour": 1,
    "non-rumor":  1,
}

# USE24-XD: categorie multi-label -> binario
#   Classe 0: qualsiasi categoria di disinformazione rilevata
#   Classe 1: "neutral" = nessuna categoria attiva
USE24_LABEL_MAP: dict[str, int] = {
    "neutral":        1,
    "sensationalism": 0,
    "conspiracy":     0,
    "hate_speech":    0,
    "satire":         0,
    "speculation":    0,
}

# -- Nomi classi per i report di classificazione sklearn ---------------------
CLASS_NAMES = ["Fake (0)", "Real (1)"]


# ##############################################################################
# ### FASE 1: DATASET E PREPARAZIONE DEI DATI                               ###
# ##############################################################################

class TextEmbeddingDataset(Dataset):
    """Dataset PyTorch per embedding testuali serializzati in Float16.

    Legge la colonna embedding_bin di un DataFrame Pandas, decodifica
    il buffer di byte in un vettore Float32 di 768 dimensioni e lo
    restituisce come tensore PyTorch insieme alla label intera.

    La deserializzazione da Float16 a Float32 avviene in __getitem__
    anziche' nel costruttore per mantenere basso il consumo di RAM
    durante il caricamento (lazy decoding).

    Args:
        dataframe:  DataFrame Pandas contenente le colonne
                    embedding_bin e target_col.
        target_col: Nome della colonna con le label numeriche intere.
    """

    def __init__(
        self,
        dataframe:  pd.DataFrame,
        target_col: str = "label_num",
    ) -> None:
        self.df         = dataframe.reset_index(drop=True)
        self.target_col = target_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Restituisce il vettore di embedding e la label del campione idx.

        Args:
            idx: Indice del campione nel DataFrame.

        Returns:
            Tupla (x, y) dove:
              x: tensore Float32 di forma [768].
              y: tensore Float32 scalare (richiesto da BCEWithLogitsLoss).
        """
        row     = self.df.iloc[idx]
        emb_raw = row["embedding_bin"]

        # Decodifica buffer di byte Float16 -> array Float32
        vec = np.frombuffer(emb_raw, dtype=np.float16).astype(np.float32)
        x   = torch.from_numpy(vec)

        y = torch.tensor(int(row[self.target_col]), dtype=torch.float32)
        return x, y


def load_and_prepare_loaders(
    parquet_path: str,
    label_map:    dict[str, int],
    splits:       dict[str, float],
    batch_size:   int            = BATCH_SIZE,
    sample_frac:  Optional[float] = None,
    random_state: int            = 42,
) -> dict[str, DataLoader]:
    """Carica un Parquet, applica il mapping delle label e crea i DataLoader.

    Funzione DRY che centralizza la logica di caricamento condivisa tra
    i tre esperimenti. Supporta split arbitrari (train/val/test o
    sottoinsiemi) e campionamento casuale per il test a freddo su USE24.

    Il parametro splits definisce le frazioni del dataset assegnate a
    ciascuna partizione. Valori supportati:
      - {"train": 0.8, "test": 0.2}               per LIAR (ablation)
      - {"train": 0.7, "val": 0.15, "test": 0.15} per PHEME
      - {"test": 1.0}                              per USE24 (inferenza)

    La mappatura delle label viene applicata dopo la normalizzazione
    testuale (lower + strip) per robustezza rispetto a variazioni di
    maiuscolo/minuscolo presenti nei diversi dataset.

    Args:
        parquet_path: Percorso del file Parquet vettorizzato FP16.
        label_map:    Dizionario da stringa testuale a intero binario.
        splits:       Dizionario {nome_split: frazione} la cui somma
                      deve essere pari a 1.0.
        batch_size:   Dimensione del mini-batch per i DataLoader.
        sample_frac:  Se specificato, campiona questa frazione del
                      dataset prima di applicare gli split. Utile per
                      il test a freddo su USE24 (sample_frac=0.20).
        random_state: Seed per riproducibilita' degli split.

    Returns:
        Dizionario {nome_split: DataLoader} con le stesse chiavi
        del parametro splits.

    Raises:
        KeyError:   Se il DataFrame non contiene la colonna status
                    o label.
        ValueError: Se nessuna riga sopravvive alla mappatura delle
                    label (label_map incompleto).
    """
    df = pd.read_parquet(parquet_path)

    # Rilevamento adattivo della colonna target
    if "status" in df.columns:
        raw_col = "status"
    elif "label" in df.columns:
        raw_col = "label"
    else:
        raise KeyError(
            "Nessuna colonna 'status' o 'label' trovata nel Parquet."
        )

    # Normalizzazione e mapping label
    df["label_num"] = (
        df[raw_col]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(label_map)
    )
    df = df.dropna(subset=["label_num"])
    df["label_num"] = df["label_num"].astype(int)

    if df.empty:
        raise ValueError(
            "Nessun record valido dopo il mapping delle label. "
            "Verificare il contenuto di label_map."
        )

    # Campionamento opzionale (es. 20% per il test a freddo su USE24)
    if sample_frac is not None:
        df, _ = train_test_split(
            df,
            train_size=sample_frac,
            random_state=random_state,
            stratify=df["label_num"],
        )

    # -- Suddivisione in split ------------------------------------------------
    loaders: dict[str, DataLoader] = {}
    split_names = list(splits.keys())

    if split_names == ["test"]:
        # Nessuna suddivisione: tutto il dataset come test
        ds = TextEmbeddingDataset(df, target_col="label_num")
        loaders["test"] = DataLoader(
            ds, batch_size=batch_size, shuffle=False
        )
        print(f"  Test : {len(ds)} campioni")
        return loaders

    # Passo 1: isola la partizione di training dal resto
    train_frac = splits.get("train", 0.0)
    remainder  = 1.0 - train_frac

    df_train, df_rest = train_test_split(
        df,
        test_size=remainder,
        random_state=random_state,
        stratify=df["label_num"],
    )

    ds_train = TextEmbeddingDataset(df_train, target_col="label_num")
    loaders["train"] = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True
    )
    print(f"  Train: {len(ds_train)} campioni")

    if "val" in split_names and "test" in split_names:
        # Passo 2: dividi il resto a meta' tra val e test
        df_val, df_test = train_test_split(
            df_rest,
            test_size=0.5,
            random_state=random_state,
            stratify=df_rest["label_num"],
        )
        ds_val  = TextEmbeddingDataset(df_val,  target_col="label_num")
        ds_test = TextEmbeddingDataset(df_test, target_col="label_num")

        loaders["val"] = DataLoader(
            ds_val, batch_size=batch_size, shuffle=False
        )
        loaders["test"] = DataLoader(
            ds_test, batch_size=batch_size, shuffle=False
        )
        print(f"  Val  : {len(ds_val)} campioni")
        print(f"  Test : {len(ds_test)} campioni")

    else:
        # Solo train + test (nessun validation set)
        ds_test = TextEmbeddingDataset(df_rest, target_col="label_num")
        loaders["test"] = DataLoader(
            ds_test, batch_size=batch_size, shuffle=False
        )
        print(f"  Test : {len(ds_test)} campioni")

    return loaders


# ##############################################################################
# ### FASE 2: ARCHITETTURA DEL MODELLO                                       ###
# ##############################################################################

class SemanticBaselineModel(nn.Module):
    """Classificatore binario Feed-Forward su embedding BERTweet [CLS].

    Architettura a tre layer lineari con Dropout, progettata per operare
    sull'embedding aggregato [CLS] prodotto da BERTweet (768 dim).
    Non accede alla sequenza completa di token: costituisce pertanto la
    baseline "solo testo" del confronto con la Bi-GCN topologica.

    Struttura:
        Linear(768 -> 256) -> ReLU -> Dropout(0.3)
        Linear(256 ->  64) -> ReLU -> Dropout(0.3)
        Linear( 64 ->   1)          [logit grezzo]

    L'ultimo strato produce un singolo logit (non una probabilita').
    La conversione in probabilita' tramite sigmoide viene eseguita
    esternamente: da BCEWithLogitsLoss durante il training (per
    stabilita' numerica) e da torch.sigmoid() durante l'inferenza.

    Args:
        input_dim:    Dimensione del vettore di input (default: 768).
        hidden_dim:   Dimensione del primo layer nascosto (default: 256).
        dropout_rate: Probabilita' di dropout (default: 0.3).
    """

    def __init__(
        self,
        input_dim:    int   = INPUT_DIM,
        hidden_dim:   int   = HIDDEN_DIM,
        dropout_rate: float = DROPOUT_RATE,
    ) -> None:
        super(SemanticBaselineModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcola il logit grezzo per un batch di embedding.

        Args:
            x: Tensore Float32 di forma [B, 768].

        Returns:
            Tensore Float32 di forma [B, 1] contenente i logit non
            normalizzati. Applicare torch.sigmoid() per ottenere
            le probabilita' di classe Real (1).
        """
        return self.network(x)


# ##############################################################################
# ### FASE 3: LOOP DI TRAINING E VALUTAZIONE                                ###
# ##############################################################################

def train_one_epoch(
    model:     SemanticBaselineModel,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device:    torch.device,
) -> float:
    """Esegue una singola epoca di addestramento sul DataLoader fornito.

    Esegue il ciclo forward -> calcolo loss -> backward -> aggiornamento
    pesi per ogni mini-batch. Il modello deve essere in modalita' train()
    prima della chiamata (il Dropout e' attivo).

    Args:
        model:     Istanza di SemanticBaselineModel in modalita' train.
        loader:    DataLoader del training set.
        criterion: Funzione di loss (BCEWithLogitsLoss).
        optimizer: Ottimizzatore Adam.
        device:    Dispositivo di calcolo (CPU o CUDA).

    Returns:
        Loss media per mini-batch sull'intera epoca.
    """
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # squeeze(1): [B, 1] -> [B] per allineamento con batch_y
        logits = model(batch_x).squeeze(1)
        loss   = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_loader(
    model:  SemanticBaselineModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Esegue l'inferenza su un DataLoader e raccoglie predizioni e label.

    Imposta il modello in modalita' eval() (Dropout disattivato) e
    disabilita il calcolo del grafo computazionale tramite no_grad()
    per minimizzare il consumo di VRAM durante la valutazione.

    La soglia di decisione e' fissa a 0.5: un logit con sigma(logit)
    > 0.5 viene classificato come Real (1), altrimenti come Fake (0).

    Args:
        model:  Istanza di SemanticBaselineModel.
        loader: DataLoader del set da valutare (val o test).
        device: Dispositivo di calcolo.

    Returns:
        Tupla (loss_media, lista_predizioni, lista_label_vere).
        loss_media e' calcolata con BCEWithLogitsLoss; le liste
        contengono interi 0/1 compatibili con scikit-learn.
    """
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    total_loss  = 0.0
    all_preds:  list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x).squeeze(1)
            loss   = criterion(logits, batch_y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_y.long().cpu().tolist())

    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels


def run_training(
    model:        SemanticBaselineModel,
    train_loader: DataLoader,
    val_loader:   Optional[DataLoader],
    device:       torch.device,
    epochs:       int   = NUM_EPOCHS,
    lr:           float = LEARNING_RATE,
) -> dict[str, list[float]]:
    """Esegue il loop di addestramento completo con validation opzionale.

    Al termine di ogni epoca stampa la training loss e, se disponibile,
    la validation loss. Il validation set e' usato esclusivamente per
    il monitoraggio: non influenza l'aggiornamento dei pesi (nessun
    early stopping implementato, per massima riproducibilita').

    Args:
        model:        Istanza di SemanticBaselineModel da addestrare.
        train_loader: DataLoader del training set.
        val_loader:   DataLoader del validation set, oppure None.
        device:       Dispositivo di calcolo.
        epochs:       Numero di epoche di addestramento.
        lr:           Learning rate per Adam.

    Returns:
        Dizionario con chiavi "train_loss" e (opzionalmente) "val_loss",
        ciascuna mappata a una lista di float (una voce per epoca).
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss":   [],
    }

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss, _, _ = evaluate_loader(
                model, val_loader, device
            )
            history["val_loss"].append(val_loss)
            print(
                f"Epoca [{epoch:02d}/{epochs}]  "
                f"Loss Train: {train_loss:.4f}  "
                f"Loss Val: {val_loss:.4f}"
            )
        else:
            print(
                f"Epoca [{epoch:02d}/{epochs}]  "
                f"Loss Train: {train_loss:.4f}"
            )

    return history


def print_evaluation_report(
    all_labels:   list[int],
    all_preds:    list[int],
    dataset_name: str,
    class_names:  list[str] = CLASS_NAMES,
) -> None:
    """Stampa accuratezza e classification report di scikit-learn.

    Args:
        all_labels:   Lista di label vere (interi 0/1).
        all_preds:    Lista di predizioni (interi 0/1).
        dataset_name: Nome del dataset per l'intestazione del report.
        class_names:  Nomi delle classi per il report sklearn.

    Returns:
        None.
    """
    acc = accuracy_score(all_labels, all_preds)
    print(
        f"\nAccuratezza su {dataset_name}: "
        f"{acc:.4f} ({acc*100:.1f}%)"
    )
    print(f"\nClassification Report — {dataset_name}:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
        )
    )


# ##############################################################################
# ### FASE 4: ENTRY POINT — ORCHESTRAZIONE DEGLI ESPERIMENTI                ###
# ##############################################################################

if __name__ == "__main__":

    print(f"Dispositivo di calcolo: {DEVICE}")
    SEP = "=" * 60

    # ------------------------------------------------------------------ #
    # ESPERIMENTO 1: Ablation Study su LIAR                              #
    # Split: 80% Train / 20% Test (nessun validation set)               #
    # Obiettivo: verifica della capacita' di generalizzazione su un      #
    # benchmark di fact-checking politico (PolitiFact, 6 classi -> 2).  #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  ESPERIMENTO 1: Ablation Study — LIAR Benchmark")
    print(SEP)

    print("Caricamento e suddivisione dataset LIAR...")
    liar_loaders = load_and_prepare_loaders(
        parquet_path = IN_LIAR,
        label_map    = LIAR_LABEL_MAP,
        splits       = {"train": 0.8, "test": 0.2},
    )

    model_liar = SemanticBaselineModel().to(DEVICE)

    print("\nAvvio addestramento su LIAR...")
    run_training(
        model        = model_liar,
        train_loader = liar_loaders["train"],
        val_loader   = None,
        device       = DEVICE,
    )

    print("\nValutazione sul test set di LIAR...")
    _, preds_liar, labels_liar = evaluate_loader(
        model_liar, liar_loaders["test"], DEVICE
    )
    print_evaluation_report(
        labels_liar, preds_liar, "LIAR (Ablation)"
    )

    # ------------------------------------------------------------------ #
    # ESPERIMENTO 2: Training e Test su PHEME (Dataset Storico 2016)    #
    # Split: 70% Train / 15% Val / 15% Test                             #
    # Obiettivo: addestramento della baseline semantica sul dataset      #
    # di riferimento storico; il modello risultante e' riutilizzato     #
    # nell'Esperimento 3 senza alcun aggiornamento dei pesi.            #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  ESPERIMENTO 2: Training su PHEME (Dati Storici 2016)")
    print(SEP)

    print("Caricamento e suddivisione dataset PHEME...")
    pheme_loaders = load_and_prepare_loaders(
        parquet_path = IN_PHEME,
        label_map    = PHEME_LABEL_MAP,
        splits       = {"train": 0.7, "val": 0.15, "test": 0.15},
    )

    # Il modello addestrato su PHEME e' riutilizzato nell'Esperimento 3
    model_pheme = SemanticBaselineModel().to(DEVICE)

    print("\nAvvio addestramento su PHEME...")
    run_training(
        model        = model_pheme,
        train_loader = pheme_loaders["train"],
        val_loader   = pheme_loaders["val"],
        device       = DEVICE,
    )

    print("\nValutazione sul test set di PHEME...")
    _, preds_pheme, labels_pheme = evaluate_loader(
        model_pheme, pheme_loaders["test"], DEVICE
    )
    print_evaluation_report(
        labels_pheme, preds_pheme, "PHEME Test Set (2016)"
    )

    # ------------------------------------------------------------------ #
    # ESPERIMENTO 3: Test a Freddo su USE24 (Concept Drift 2016->2024)  #
    # Il modello addestrato su PHEME viene valutato senza alcun          #
    # fine-tuning su un campione casuale stratificato del 20% di USE24. #
    # Nessun aggiornamento dei pesi: e' una valutazione di inferenza    #
    # pura per misurare la degradazione delle prestazioni causata dal   #
    # Concept Drift linguistico e topologico tra il 2016 e il 2024.     #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  ESPERIMENTO 3: Test a Freddo su USE24 (Concept Drift)")
    print(SEP)

    print("Caricamento campione USE24 (20% stratificato)...")
    use24_loaders = load_and_prepare_loaders(
        parquet_path = IN_USE24,
        label_map    = USE24_LABEL_MAP,
        splits       = {"test": 1.0},
        sample_frac  = 0.20,
    )

    print(
        "\nValutazione a freddo — "
        "modello PHEME 2016 su USE24 2024..."
    )
    _, preds_use24, labels_use24 = evaluate_loader(
        model_pheme, use24_loaders["test"], DEVICE
    )
    print_evaluation_report(
        labels_use24,
        preds_use24,
        "USE24 — Concept Drift 2016 -> 2024",
    )

    print(f"\n{SEP}")
    print("  Pipeline Baseline Semantica completata.")
    print(SEP)
