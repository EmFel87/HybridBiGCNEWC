# =============================================================================
# FILE:    baseline_topological.py
# SCOPO:   Baseline topologica per il rilevamento delle Fake News tramite
#          Bidirectional Graph Convolutional Network (Bi-GCN).
#          Il modulo esegue due esperimenti in sequenza:
#            1. Training + Validazione + Test su PHEME (dati storici 2016).
#            2. Test a freddo su USE24 (misurazione del Concept Drift 2024).
#          L'architettura Bi-GCN opera su grafi di propagazione pre-calcolati
#          (file .pt, output di feature_extraction_topological.py), dove ogni
#          nodo e' rappresentato dall'embedding [CLS] BERTweet a 768 dimensioni
#          e gli archi codificano la struttura della cascata di risposta.
# DIPENDENZE: torch, torch_geometric, scikit-learn, pandas
# MODULO:  Baseline Topologica — esperimento indipendente dai moduli ETL
# =============================================================================


# ##############################################################################
# ### FASE 0: IMPORT E COSTANTI GLOBALI                                      ###
# ##############################################################################

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import Optional

# -- Dispositivo di calcolo ---------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Percorsi Google Drive ----------------------------------------------------
DRIVE_BASE        = "/content/drive/MyDrive/Tesi"
IN_PHEME_PARQUET  = f"{DRIVE_BASE}/Parquet_Finali/PHEME_Vectorized_FP16.parquet"
IN_PHEME_GRAPHS   = f"{DRIVE_BASE}/Grafi_PyG/PHEME_graphs_list.pt"
IN_USE24_PARQUET  = f"{DRIVE_BASE}/Parquet_Finali/USE24_Vectorized_FP16.parquet"
IN_USE24_GRAPHS   = f"{DRIVE_BASE}/Grafi_PyG/USE24_graphs_list.pt"

# -- Iperparametri del modello ------------------------------------------------
IN_FEATS      = 768   # Dimensione embedding [CLS] BERTweet per nodo
HIDDEN_FEATS  = 64    # Dimensione degli hidden state GCN
DROPOUT_RATE  = 0.3
LEARNING_RATE = 5e-4  # Valore ridotto rispetto alla baseline semantica per
                      # garantire convergenza stabile su strutture a grafo
BATCH_SIZE    = 32
NUM_EPOCHS    = 15    # Epoche aumentate per consentire la propagazione dei
                      # messaggi su cascate di profondita' variabile

# -- Mapping label testuale -> classe binaria (0 = Fake, 1 = Real) -----------
#    I dizionari replicano il LABEL_MAP di feature_extraction_topological.py,
#    garantendo coerenza tra la fase di costruzione dei grafi e quella
#    di addestramento. Le varianti ortografiche britanniche/americane
#    sono incluse per copertura completa del vocabolario PHEME.

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

# -- Nomi classi per i report di classificazione sklearn ---------------------
CLASS_NAMES = ["Fake/Rumor (0)", "Real/Non-Rumor (1)"]


# ##############################################################################
# ### FASE 1: DATA PREPARATION                                               ###
# ##############################################################################

def load_and_map_graphs(
    parquet_path: str,
    pt_path:      str,
    label_map:    dict[str, int],
) -> tuple[list, list[int]]:
    """Carica i grafi PyG e inietta le label numeriche tramite join con il Parquet.

    Il file .pt prodotto da feature_extraction_topological.py contiene
    grafi con label y = -1 (valore sentinella), perche' al momento della
    costruzione topologica le etichette non erano ancora state associate
    agli oggetti Data. Questa funzione esegue il join tra i grafi e il
    Parquet vettorizzato usando root_id (lato grafi) e node_id (lato
    Parquet) come chiave di collegamento.

    Il meccanismo di join tramite dizionario Python {node_id: label_num}
    garantisce complessita' O(1) per ogni lookup, rendendo l'operazione
    efficiente anche su dataset con oltre 100.000 nodi (USE24).

    Args:
        parquet_path: Percorso del Parquet vettorizzato FP16 contenente
                      le colonne node_id e status (o label).
        pt_path:      Percorso del file .pt contenente la lista di grafi
                      Data prodotta da feature_extraction_topological.py.
        label_map:    Dizionario da stringa testuale a intero binario
                      (0 = Fake, 1 = Real). Deve coprire tutte le label
                      testuali presenti nella colonna status del Parquet.

    Returns:
        Tupla (valid_graphs, labels) dove:
          valid_graphs: lista di oggetti Data con g.y aggiornato a
                        torch.tensor([label], dtype=torch.float32).
          labels:       lista parallela di interi (0/1) per lo stratify
                        negli split scikit-learn.

    Raises:
        KeyError: Se il Parquet non contiene le colonne node_id/status.
    """
    # -- Costruzione del dizionario di lookup label --------------------------
    df = pd.read_parquet(parquet_path)

    raw_col = "status" if "status" in df.columns else "label"
    id_col  = "node_id" if "node_id" in df.columns else "tweet_id"

    df["label_num"] = (
        df[raw_col]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(label_map)
    )
    df = df.dropna(subset=["label_num"])
    df["label_num"] = df["label_num"].astype(int)

    label_dict: dict[str, int] = dict(
        zip(df[id_col].astype(str), df["label_num"])
    )
    print(
        f"Dizionario label costruito: {len(label_dict)} voci valide."
    )

    # -- Caricamento grafi e iniezione label ---------------------------------
    # weights_only=False: richiesto da PyTorch >= 2.6 per oggetti
    # personalizzati come torch_geometric.data.Data, che non sono
    # tensori semplici e non superano il filtro di sicurezza di default.
    graphs = torch.load(pt_path, weights_only=False)
    print(f"Grafi caricati dal file .pt: {len(graphs)}.")

    valid_graphs: list  = []
    labels:       list[int] = []

    for g in graphs:
        root_id = str(g.root_id)
        if root_id in label_dict:
            label_val = label_dict[root_id]
            g.y = torch.tensor([label_val], dtype=torch.float32)
            valid_graphs.append(g)
            labels.append(label_val)

    print(
        f"Grafi con label valida (join riuscito): {len(valid_graphs)}."
    )
    return valid_graphs, labels


def create_graph_loaders(
    graphs:       list,
    labels:       list[int],
    splits:       dict[str, float],
    batch_size:   int            = BATCH_SIZE,
    sample_frac:  Optional[float] = None,
    random_state: int            = 42,
) -> dict[str, DataLoader]:
    """Crea i DataLoader PyG a partire da una lista di grafi.

    Supporta split arbitrari e campionamento stratificato opzionale per
    il test a freddo. Il DataLoader di PyTorch Geometric combina grafi
    di dimensioni eterogenee in un unico "super-grafo" sparso a blocchi
    diagonali, consentendo il calcolo parallelo su GPU anche quando le
    conversazioni hanno numero di nodi variabile.

    Args:
        graphs:       Lista di oggetti Data con g.y gia' impostato.
        labels:       Lista parallela di interi per lo stratify.
        splits:       Dizionario {nome_split: frazione}. Esempi:
                        {"train": 0.7, "val": 0.15, "test": 0.15}
                        {"test": 1.0} per inferenza pura.
        batch_size:   Dimensione del mini-batch.
        sample_frac:  Se specificato, campiona questa frazione prima
                      degli split (es. 0.20 per il test USE24).
        random_state: Seed per riproducibilita'.

    Returns:
        Dizionario {nome_split: DataLoader} con le chiavi di splits.
    """
    # Campionamento opzionale pre-split
    if sample_frac is not None:
        graphs, _, labels, _ = train_test_split(
            graphs,
            labels,
            train_size   = sample_frac,
            random_state = random_state,
            stratify     = labels,
        )
        print(
            f"Campionamento {sample_frac*100:.0f}%: "
            f"{len(graphs)} grafi selezionati."
        )

    loaders:     dict[str, DataLoader] = {}
    split_names: list[str] = list(splits.keys())

    if split_names == ["test"]:
        loaders["test"] = DataLoader(
            graphs, batch_size=batch_size, shuffle=False
        )
        print(f"  Test : {len(graphs)} grafi")
        return loaders

    # Passo 1: isolamento del training set
    train_frac = splits.get("train", 0.0)
    (
        train_g, rest_g,
        train_l, rest_l,
    ) = train_test_split(
        graphs,
        labels,
        test_size    = 1.0 - train_frac,
        random_state = random_state,
        stratify     = labels,
    )

    loaders["train"] = DataLoader(
        train_g, batch_size=batch_size, shuffle=True
    )
    print(f"  Train: {len(train_g)} grafi")

    if "val" in split_names and "test" in split_names:
        # Passo 2: divisione a meta' tra val e test
        val_g, test_g, _, _ = train_test_split(
            rest_g,
            rest_l,
            test_size    = 0.5,
            random_state = random_state,
            stratify     = rest_l,
        )
        loaders["val"]  = DataLoader(
            val_g, batch_size=batch_size, shuffle=False
        )
        loaders["test"] = DataLoader(
            test_g, batch_size=batch_size, shuffle=False
        )
        print(f"  Val  : {len(val_g)} grafi")
        print(f"  Test : {len(test_g)} grafi")

    else:
        loaders["test"] = DataLoader(
            rest_g, batch_size=batch_size, shuffle=False
        )
        print(f"  Test : {len(rest_g)} grafi")

    return loaders


# ##############################################################################
# ### FASE 2: ARCHITETTURA DEL MODELLO                                       ###
# ##############################################################################

class BiGCNBaselineModel(nn.Module):
    """Rete Bi-GCN per la classificazione binaria di cascate di propagazione.

    Implementa l'architettura Bidirectional Graph Convolutional Network
    proposta da Bian et al. (2020) "Rumor Detection on Social Media with
    Bi-Directional Graph Convolutional Networks". Il modello processa
    ogni cascata attraverso due rami GCN paralleli e indipendenti:

    Ramo Top-Down (TD):
        Modella il flusso dell'informazione dalla radice verso le foglie.
        Cattura il pattern di diffusione della notizia originale.
        Usa edge_index nativo del grafo (parent -> child).

    Ramo Bottom-Up (BU):
        Modella il flusso di feedback dalle foglie verso la radice.
        Cattura la risposta aggregata dell'audience alla notizia.
        Usa edge_index invertito tramite flip(0) (child -> parent).

    L'inversione tramite .flip(0) e' equivalente alla trasposta della
    matrice di adiacenza: [src, dst] -> [dst, src]. Questa operazione
    in-place e' computazionalmente trascurabile rispetto alla convoluzione
    e non richiede la memorizzazione di un secondo edge_index.

    I vettori di grafo TD e BU vengono concatenati e proiettati tramite
    un classificatore Feed-Forward a due layer verso un logit binario.

    Struttura:
        [TD]  GCNConv(768->64) -> ReLU -> GCNConv(64->64) -> MeanPool
        [BU]  GCNConv(768->64) -> ReLU -> GCNConv(64->64) -> MeanPool
        [FC]  Linear(128->64) -> ReLU -> Dropout -> Linear(64->1)

    Args:
        in_feats:     Dimensione delle feature per nodo (default: 768).
        hidden_feats: Dimensione degli hidden state GCN (default: 64).
        dropout_rate: Probabilita' di dropout nel classificatore.
    """

    def __init__(
        self,
        in_feats:     int   = IN_FEATS,
        hidden_feats: int   = HIDDEN_FEATS,
        dropout_rate: float = DROPOUT_RATE,
    ) -> None:
        super(BiGCNBaselineModel, self).__init__()

        # -- Ramo Top-Down ---------------------------------------------------
        self.td_conv1 = GCNConv(in_feats, hidden_feats)
        self.td_conv2 = GCNConv(hidden_feats, hidden_feats)

        # -- Ramo Bottom-Up --------------------------------------------------
        self.bu_conv1 = GCNConv(in_feats, hidden_feats)
        self.bu_conv2 = GCNConv(hidden_feats, hidden_feats)

        # -- Classificatore finale -------------------------------------------
        # Input: concatenazione TD || BU -> hidden_feats * 2
        self.fc1     = nn.Linear(hidden_feats * 2, hidden_feats)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2     = nn.Linear(hidden_feats, 1)

    def forward(self, data) -> torch.Tensor:
        """Calcola il logit binario per un batch di grafi.

        Args:
            data: Batch PyG (oggetto Batch) contenente gli attributi:
                  x          — feature matrix dei nodi [N_tot, 768]
                  edge_index — archi top-down [2, E_tot]
                  batch      — vettore di assegnazione nodo->grafo [N_tot]

        Returns:
            Tensore Float32 di forma [B, 1] con i logit non normalizzati.
            Applicare torch.sigmoid() per ottenere le probabilita' di
            classe Real (1).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Inversione degli archi per il ramo Bottom-Up:
        # flip(0) scambia le righe [src, dst] -> [dst, src]
        edge_index_bu = edge_index.flip(0)

        # -- Propagazione Top-Down -------------------------------------------
        x_td = F.relu(self.td_conv1(x, edge_index))
        x_td = F.relu(self.td_conv2(x_td, edge_index))
        g_td = global_mean_pool(x_td, batch)  # [B, hidden_feats]

        # -- Propagazione Bottom-Up ------------------------------------------
        x_bu = F.relu(self.bu_conv1(x, edge_index_bu))
        x_bu = F.relu(self.bu_conv2(x_bu, edge_index_bu))
        g_bu = global_mean_pool(x_bu, batch)  # [B, hidden_feats]

        # -- Fusione e classificazione ---------------------------------------
        g_fused = torch.cat([g_td, g_bu], dim=1)  # [B, hidden_feats*2]
        out     = F.relu(self.fc1(g_fused))
        out     = self.dropout(out)
        out     = self.fc2(out)                    # [B, 1]

        return out


# ##############################################################################
# ### FASE 3: LOOP DI TRAINING E VALUTAZIONE                                ###
# ##############################################################################

def train_one_epoch_graph(
    model:     BiGCNBaselineModel,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device:    torch.device,
) -> float:
    """Esegue una singola epoca di addestramento su grafi PyG.

    Itera sui mini-batch del DataLoader, dove ogni batch e' un super-grafo
    sparso a blocchi diagonali che unisce i singoli grafi della cascata.
    Il vettore data.batch mappa ogni nodo al grafo di appartenenza e
    viene usato da global_mean_pool per aggregare i nodi per grafo.

    Args:
        model:     Istanza di BiGCNBaselineModel in modalita' train.
        loader:    DataLoader PyG del training set.
        criterion: Funzione di loss (BCEWithLogitsLoss).
        optimizer: Ottimizzatore Adam.
        device:    Dispositivo di calcolo (CPU o CUDA).

    Returns:
        Loss media per mini-batch sull'intera epoca.
    """
    model.train()
    total_loss = 0.0

    for batch_data in loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()

        logits = model(batch_data)

        # squeeze(): [B, 1] -> [B]; la y dei grafi e' gia' [B, 1],
        # quindi entrambi vengono appiattiti per BCEWithLogitsLoss.
        loss = criterion(
            logits.squeeze(), batch_data.y.squeeze()
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_graph_loader(
    model:  BiGCNBaselineModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Esegue l'inferenza su un DataLoader PyG e raccoglie predizioni e label.

    Gestisce esplicitamente il caso limite in cui un mini-batch contiene
    un solo grafo: dopo squeeze(), il tensore perde la dimensione batch
    diventando uno scalare (dim=0). In questo caso, unsqueeze(0) ripristina
    la dimensione [1] necessaria per l'extend della lista di output.

    Args:
        model:  Istanza di BiGCNBaselineModel.
        loader: DataLoader PyG del set di valutazione.
        device: Dispositivo di calcolo.

    Returns:
        Tupla (loss_media, lista_predizioni, lista_label_vere).
        Le liste contengono interi 0/1 compatibili con scikit-learn.
    """
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    total_loss  = 0.0
    all_preds:  list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)

            logits = model(batch_data).squeeze()
            y_true = batch_data.y.squeeze()

            # Gestione batch singolo: squeeze() su tensore [1, 1]
            # produce uno scalare (dim=0); unsqueeze(0) lo ripristina a [1]
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            if y_true.dim() == 0:
                y_true = y_true.unsqueeze(0)

            loss = criterion(logits, y_true)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_true.long().cpu().tolist())

    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels


def run_graph_training(
    model:        BiGCNBaselineModel,
    train_loader: DataLoader,
    val_loader:   Optional[DataLoader],
    device:       torch.device,
    epochs:       int   = NUM_EPOCHS,
    lr:           float = LEARNING_RATE,
) -> dict[str, list[float]]:
    """Esegue il loop di addestramento completo con validation opzionale.

    Il learning rate ridotto (5e-4 vs 1e-3 della baseline semantica) e'
    motivato dalla natura del segnale topologico: le convoluzioni GCN su
    grafi con profondita' e branching factor variabili producono gradienti
    con varianza elevata nelle prime epoche, che un lr troppo alto ampli-
    ficherebbe fino all'instabilita' del training.

    Args:
        model:        Istanza di BiGCNBaselineModel da addestrare.
        train_loader: DataLoader PyG del training set.
        val_loader:   DataLoader PyG del validation set, oppure None.
        device:       Dispositivo di calcolo.
        epochs:       Numero di epoche di addestramento.
        lr:           Learning rate per Adam.

    Returns:
        Dizionario con chiavi "train_loss" e "val_loss", ciascuna
        mappata a una lista di float (una voce per epoca).
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss":   [],
    }

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_graph(
            model, train_loader, criterion, optimizer, device
        )
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss, _, _ = evaluate_graph_loader(
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


def print_graph_evaluation_report(
    all_labels:   list[int],
    all_preds:    list[int],
    dataset_name: str,
    class_names:  list[str] = CLASS_NAMES,
) -> None:
    """Stampa accuratezza e classification report scikit-learn per grafi.

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
    # ESPERIMENTO 1: Training e Test Topologico su PHEME (2016)         #
    # Split: 70% Train / 15% Val / 15% Test                             #
    # Obiettivo: addestrare la Bi-GCN sulla struttura delle cascate di  #
    # propagazione del dataset storico; il modello risultante e'         #
    # trasferito direttamente all'Esperimento 2 senza fine-tuning.      #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  ESPERIMENTO 1: Baseline Topologica — PHEME (2016)")
    print(SEP)

    print("\nCaricamento e join label — PHEME...")
    pheme_graphs, pheme_labels = load_and_map_graphs(
        parquet_path = IN_PHEME_PARQUET,
        pt_path      = IN_PHEME_GRAPHS,
        label_map    = PHEME_LABEL_MAP,
    )

    print("\nCreazione DataLoader PHEME (70 / 15 / 15)...")
    pheme_loaders = create_graph_loaders(
        graphs  = pheme_graphs,
        labels  = pheme_labels,
        splits  = {"train": 0.7, "val": 0.15, "test": 0.15},
    )

    # Il modello addestrato su PHEME e' riutilizzato nell'Esperimento 2
    model_topo = BiGCNBaselineModel(
        in_feats     = IN_FEATS,
        hidden_feats = HIDDEN_FEATS,
    ).to(DEVICE)

    print("\nAvvio addestramento Bi-GCN su PHEME...")
    run_graph_training(
        model        = model_topo,
        train_loader = pheme_loaders["train"],
        val_loader   = pheme_loaders["val"],
        device       = DEVICE,
    )

    print("\nValutazione sul test set di PHEME...")
    _, preds_pheme, labels_pheme = evaluate_graph_loader(
        model_topo, pheme_loaders["test"], DEVICE
    )
    print_graph_evaluation_report(
        labels_pheme, preds_pheme, "PHEME Test Set (2016)"
    )

    # ------------------------------------------------------------------ #
    # ESPERIMENTO 2: Test a Freddo su USE24 (Concept Drift 2016->2024)  #
    # Il modello Bi-GCN addestrato su PHEME viene valutato senza alcun  #
    # aggiornamento dei pesi su un campione stratificato del 20% di     #
    # USE24. L'assenza di struttura gerarchica in USE24 (grafi a nodo   #
    # singolo) costituisce il "Drift Strutturale" documentato in tesi:  #
    # il modello non puo' sfruttare il segnale di propagazione e deve   #
    # contare unicamente sulle feature BERTweet dei singoli nodi.       #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  ESPERIMENTO 2: Test a Freddo su USE24 (Concept Drift)")
    print(SEP)

    print("\nCaricamento e join label — USE24...")
    use24_graphs, use24_labels = load_and_map_graphs(
        parquet_path = IN_USE24_PARQUET,
        pt_path      = IN_USE24_GRAPHS,
        label_map    = USE24_LABEL_MAP,
    )

    print("\nCreazione DataLoader USE24 (campione 20%)...")
    use24_loaders = create_graph_loaders(
        graphs       = use24_graphs,
        labels       = use24_labels,
        splits       = {"test": 1.0},
        sample_frac  = 0.20,
    )

    print(
        "\nValutazione a freddo — "
        "modello PHEME 2016 su USE24 2024..."
    )
    _, preds_use24, labels_use24 = evaluate_graph_loader(
        model_topo, use24_loaders["test"], DEVICE
    )
    print_graph_evaluation_report(
        labels_use24,
        preds_use24,
        "USE24 — Concept Drift Topologico 2016 -> 2024",
    )

    print(f"\n{SEP}")
    print("  Pipeline Baseline Topologica completata.")
    print(SEP)
