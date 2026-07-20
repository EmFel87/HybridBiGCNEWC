# =============================================================================
# FILE:    hybrid_ewc_model.py
# SCOPO:   Modello ibrido Gated Bi-GCN con Continual Learning tramite
#          Elastic Weight Consolidation (EWC) per il rilevamento delle
#          Fake News in presenza di Concept Drift temporale (2016->2024).
#          Struttura del file:
#            FASE 0 — Import e costanti globali
#            FASE 1 — Data preparation (caricamento, join label, split)
#            FASE 2 — Architettura HybridGatedBiGCN
#            FASE 3 — EWC: Fisher Information Matrix e penale
#            FASE 4 — Loop di training, fine-tuning e valutazione
#            FASE 5 — Entry point (orchestrazione completa)
# DIPENDENZE: torch, torch_geometric, scikit-learn, pandas
# MODULO:  Modello ibrido principale — richiede i Parquet e i .pt
#          prodotti da feature_extraction_semantic.py e
#          feature_extraction_topological.py
# =============================================================================


# ##############################################################################
# ### FASE 0: IMPORT E COSTANTI GLOBALI                                      ###
# ##############################################################################

import copy
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
DRIVE_BASE       = "/content/drive/MyDrive/Tesi"
IN_PHEME_PARQUET = (
    f"{DRIVE_BASE}/Parquet_Finali/PHEME_Vectorized_FP16.parquet"
)
IN_PHEME_GRAPHS  = f"{DRIVE_BASE}/Grafi_PyG/PHEME_graphs_list.pt"
IN_USE24_PARQUET = (
    f"{DRIVE_BASE}/Parquet_Finali/USE24_Vectorized_FP16.parquet"
)
IN_USE24_GRAPHS  = f"{DRIVE_BASE}/Grafi_PyG/USE24_graphs_list.pt"

# -- Iperparametri architettura -----------------------------------------------
IN_FEATS     = 768   # Dimensione embedding [CLS] BERTweet per nodo
HIDDEN_FEATS = 64    # Dimensione hidden state convoluzioni GCN
COMMON_DIM   = 128   # Dimensione spazio comune di fusione testo/grafo
DROPOUT_RATE = 0.3

# -- Iperparametri training principale (PHEME) --------------------------------
LR_MAIN    = 5e-4
WD_MAIN    = 1e-4    # Weight decay AdamW
EPOCHS_MAIN = 15
BATCH_SIZE  = 32
CLIP_NORM   = 1.0    # Soglia gradient clipping (stabilita' su grafi profondi)

# -- Iperparametri fine-tuning (EWC e Naive, USE24) ---------------------------
LR_FT      = 5e-4
WD_FT      = 1e-4
EPOCHS_FT  = 5

# -- Coefficiente EWC ---------------------------------------------------------
#    LAMBDA_EWC bilancia loss sul task 2024 e penale sulla memoria 2016.
#    Valore selezionato empiricamente: protegge i pesi critici mantenendo
#    la degradazione su PHEME entro 5 punti percentuali di accuratezza.
LAMBDA_EWC = 5000

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

CLASS_NAMES = ["Fake/Rumor (0)", "Real/Non-Rumor (1)"]


# ##############################################################################
# ### FASE 1: DATA PREPARATION                                               ###
# ##############################################################################

def load_and_map_graphs(
    parquet_path: str,
    pt_path:      str,
    label_map:    dict[str, int],
) -> tuple[list, list[int]]:
    """Carica i grafi PyG e inietta le label tramite join con il Parquet.

    I grafi nel file .pt hanno label y = -1 (valore sentinella assegnato
    in feature_extraction_topological.py al momento della costruzione
    topologica, prima che le annotazioni fossero disponibili). Questa
    funzione risolve il join usando root_id (lato grafi) e node_id
    (lato Parquet) come chiave di collegamento.

    Il dizionario Python {node_id: label_num} garantisce lookup O(1)
    per grafo, rendendo l'operazione efficiente su dataset con centinaia
    di migliaia di nodi come USE24.

    Args:
        parquet_path: Percorso del Parquet vettorizzato FP16 prodotto
                      da feature_extraction_semantic.py.
        pt_path:      Percorso del file .pt con la lista di oggetti
                      Data prodotta da feature_extraction_topological.py.
        label_map:    Dizionario {stringa_testuale: intero_binario}.
                      Deve coprire tutti i valori presenti nella colonna
                      status (o label) del Parquet.

    Returns:
        Tupla (valid_graphs, labels) dove:
          valid_graphs: lista di Data con g.y = tensor([label], float32).
          labels:       lista parallela di int per lo stratify sklearn.

    Raises:
        KeyError: Se il Parquet non contiene le colonne node_id/status.
    """
    df      = pd.read_parquet(parquet_path)
    raw_col = "status" if "status" in df.columns else "label"
    id_col  = "node_id" if "node_id" in df.columns else "tweet_id"

    df["label_num"] = (
        df[raw_col]
        .astype(str).str.lower().str.strip()
        .map(label_map)
    )
    df = df.dropna(subset=["label_num"])
    df["label_num"] = df["label_num"].astype(int)

    label_dict: dict[str, int] = dict(
        zip(df[id_col].astype(str), df["label_num"])
    )
    print(f"Dizionario label costruito: {len(label_dict)} voci.")

    # weights_only=False richiesto da PyTorch >= 2.6 per oggetti
    # personalizzati come torch_geometric.data.Data
    graphs = torch.load(pt_path, weights_only=False)
    print(f"Grafi caricati dal file .pt: {len(graphs)}.")

    valid_graphs: list     = []
    labels:       list[int] = []

    for g in graphs:
        root_id = str(g.root_id)
        if root_id in label_dict:
            lv      = label_dict[root_id]
            g.y     = torch.tensor([lv], dtype=torch.float32)
            valid_graphs.append(g)
            labels.append(lv)

    print(f"Grafi con label valida (join OK): {len(valid_graphs)}.")
    return valid_graphs, labels


def create_graph_loaders(
    graphs:       list,
    labels:       list[int],
    splits:       dict[str, float],
    batch_size:   int             = BATCH_SIZE,
    sample_frac:  Optional[float] = None,
    random_state: int             = 42,
) -> dict[str, DataLoader]:
    """Crea i DataLoader PyG da una lista di grafi con split configurabile.

    Il DataLoader di PyTorch Geometric combina grafi di dimensioni
    eterogenee in un super-grafo sparso a blocchi diagonali, permettendo
    la convoluzione GCN su batch con conversazioni di lunghezza variabile
    in un unico forward pass su GPU.

    Configurazioni di split supportate:
      {"train": 0.7, "val": 0.15, "test": 0.15}  — con validation
      {"train": 0.8, "test": 0.2}                 — senza validation
      {"test": 1.0}                               — solo inferenza

    Args:
        graphs:       Lista di oggetti Data con g.y impostato.
        labels:       Lista parallela di int per lo stratify.
        splits:       Dizionario {nome_split: frazione}.
        batch_size:   Dimensione del mini-batch PyG.
        sample_frac:  Se specificato, campiona questa frazione prima
                      degli split (es. 0.20 per il cold test USE24).
        random_state: Seed per riproducibilita'.

    Returns:
        Dizionario {nome_split: DataLoader} con le chiavi di splits.
    """
    if sample_frac is not None:
        graphs, _, labels, _ = train_test_split(
            graphs, labels,
            train_size   = sample_frac,
            random_state = random_state,
            stratify     = labels,
        )
        print(
            f"Campionamento {sample_frac*100:.0f}%: "
            f"{len(graphs)} grafi selezionati."
        )

    loaders:     dict[str, DataLoader] = {}
    split_names: list[str]             = list(splits.keys())

    # Caso: solo set di test (inferenza pura, nessun training)
    if split_names == ["test"]:
        loaders["test"] = DataLoader(
            graphs, batch_size=batch_size, shuffle=False
        )
        print(f"  Test : {len(graphs)} grafi")
        return loaders

    # Passo 1: isolamento del training set
    train_frac = splits.get("train", 0.0)
    train_g, rest_g, train_l, rest_l = train_test_split(
        graphs, labels,
        test_size    = 1.0 - train_frac,
        random_state = random_state,
        stratify     = labels,
    )
    loaders["train"] = DataLoader(
        train_g, batch_size=batch_size, shuffle=True
    )
    print(f"  Train: {len(train_g)} grafi")

    # Passo 2: suddivisione del resto tra val e test (50/50)
    if "val" in split_names and "test" in split_names:
        val_g, test_g, _, _ = train_test_split(
            rest_g, rest_l,
            test_size    = 0.5,
            random_state = random_state,
            stratify     = rest_l,
        )
        loaders["val"]  = DataLoader(
            val_g,  batch_size=batch_size, shuffle=False
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

class HybridGatedBiGCN(nn.Module):
    """Classificatore ibrido Gated Bi-GCN per cascate di propagazione.

    Fonde un ramo topologico Bi-GCN con un ramo semantico BERTweet
    tramite un modulo di gating adattivo per componente. L'architettura
    risolve il disallineamento strutturale tra PHEME (alberi gerarchici,
    segnale topologico forte) e USE24 (nodi isolati, segnale topologico
    nullo): il gate apprende autonomamente a pesare i due contributi in
    base alla ricchezza della struttura del grafo.

    Comportamento atteso del gate:
      alpha -> 1 per USE24: nodi isolati, il ramo testuale domina.
      alpha -> 0 per PHEME: alberi profondi, la topologia domina.

    Struttura interna:
      [TD]   GCNConv(768->64,GELU) x2 -> MeanPool
      [BU]   GCNConv(768->64,GELU) x2 -> MeanPool  [flip(0)]
      [Proj_topo] Linear(128->128) -> LayerNorm -> GELU
      [Proj_sem]  Linear(768->128) -> LayerNorm -> GELU
      [Gate]      Linear(256->128) -> Sigmoid
      [Fusion]    alpha * z_sem + (1 - alpha) * z_topo
      [FC]        Linear(128->64,GELU) -> Dropout -> Linear(64->1)

    GELU e' usato in tutto il modello (incluso il classificatore
    finale) per mantenere gradienti non nulli sulle attivazioni
    leggermente negative: proprieta' critica per il calcolo accurato
    della Fisher Information Matrix durante la fase EWC.

    Args:
        in_feats:     Feature per nodo da BERTweet (default: 768).
        hidden_feats: Dimensione hidden GCN (default: 64).
        common_dim:   Spazio comune di fusione (default: 128).
        dropout_rate: Probabilita' di dropout (default: 0.3).
    """

    def __init__(
        self,
        in_feats:     int   = IN_FEATS,
        hidden_feats: int   = HIDDEN_FEATS,
        common_dim:   int   = COMMON_DIM,
        dropout_rate: float = DROPOUT_RATE,
    ) -> None:
        super(HybridGatedBiGCN, self).__init__()

        # -- Ramo topologico Top-Down ----------------------------------------
        self.td_conv1 = GCNConv(in_feats, hidden_feats)
        self.td_conv2 = GCNConv(hidden_feats, hidden_feats)

        # -- Ramo topologico Bottom-Up ----------------------------------------
        self.bu_conv1 = GCNConv(in_feats, hidden_feats)
        self.bu_conv2 = GCNConv(hidden_feats, hidden_feats)

        # -- Proiezione topologica: hidden_feats*2 -> common_dim -------------
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_feats * 2, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
        )

        # -- Proiezione semantica: in_feats -> common_dim --------------------
        self.text_proj = nn.Sequential(
            nn.Linear(in_feats, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
        )

        # -- Modulo di fusione adattiva (Gate) --------------------------------
        # Riceve z_sem || z_topo (common_dim * 2) e produce alpha in (0,1)
        self.gate = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
            nn.Sigmoid(),
        )

        # -- Classificatore finale -------------------------------------------
        self.fc1     = nn.Linear(common_dim, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2     = nn.Linear(64, 1)

    def forward(self, data) -> torch.Tensor:
        """Calcola il logit binario per un batch di grafi.

        Args:
            data: Batch PyG con attributi:
                  x          — feature nodi [N_tot, in_feats]
                  edge_index — archi top-down [2, E_tot]
                  batch      — assegnazione nodo->grafo [N_tot]
                  ptr        — indici di inizio di ogni grafo nel batch

        Returns:
            Tensore Float32 [B, 1] con logit non normalizzati.
            Applicare torch.sigmoid() per le probabilita' di classe.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Inversione archi per ramo Bottom-Up:
        # flip(0) equivale alla trasposta della matrice di adiacenza
        edge_index_bu = edge_index.flip(0)

        # -- Propagazione Top-Down -------------------------------------------
        x_td = F.gelu(self.td_conv1(x, edge_index))
        x_td = F.gelu(self.td_conv2(x_td, edge_index))
        g_td = global_mean_pool(x_td, batch)  # [B, hidden_feats]

        # -- Propagazione Bottom-Up ------------------------------------------
        x_bu = F.gelu(self.bu_conv1(x, edge_index_bu))
        x_bu = F.gelu(self.bu_conv2(x_bu, edge_index_bu))
        g_bu = global_mean_pool(x_bu, batch)  # [B, hidden_feats]

        # -- Proiezione topologica -------------------------------------------
        g_topo = torch.cat([g_td, g_bu], dim=1)  # [B, hidden_feats*2]
        z_topo = self.graph_proj(g_topo)          # [B, common_dim]

        # -- Estrazione embedding nodo radice (ramo semantico) ---------------
        # data.ptr[:-1] contiene l'indice del primo nodo di ogni grafo
        # nel super-grafo batch, corrispondente al tweet radice.
        root_idx = data.ptr[:-1]
        x_root   = x[root_idx]          # [B, in_feats]
        z_sem    = self.text_proj(x_root) # [B, common_dim]

        # -- Fusione gated ---------------------------------------------------
        z_cat  = torch.cat([z_sem, z_topo], dim=1)  # [B, common_dim*2]
        alpha  = self.gate(z_cat)                    # [B, common_dim]

        # Combinazione convessa per componente dello spazio comune
        h = alpha * z_sem + (1.0 - alpha) * z_topo  # [B, common_dim]

        # -- Classificazione -------------------------------------------------
        out = F.gelu(self.fc1(h))
        out = self.dropout(out)
        return self.fc2(out)                         # [B, 1]


# ##############################################################################
# ### FASE 3: EWC E CONTINUAL LEARNING                                       ###
# ##############################################################################

def compute_fisher_matrix(
    model:     HybridGatedBiGCN,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Calcola la Fisher Information Matrix diagonale sul task storico.

    La Fisher Information Matrix (FIM) misura la curvatura della loss
    rispetto a ciascun parametro: parametri con FIM alta sono "importanti"
    per il task corrente (PHEME 2016) e verranno vincolati da EWC durante
    il fine-tuning su USE24.

    Approssimazione diagonale adottata (Kirkpatrick et al., 2017):
        F_i ≈ E[ ( d/d theta_i  log p(y | x, theta) )^2 ]

    In pratica, F_i viene stimata come media del quadrato dei gradienti
    su tutti i mini-batch del training set storico. Questa formulazione
    e' computazionalmente trattabile su reti con milioni di parametri ed
    e' equivalente al metodo "online EWC" della letteratura.

    La funzione salva anche il clone dei pesi ottimali theta* al momento
    della chiamata: questi valori costituiscono il punto di riferimento
    intorno al quale la penale EWC penalizzera' i futuri scostamenti.

    Args:
        model:     HybridGatedBiGCN dopo il training su PHEME. Deve
                   essere in modalita' eval() prima della chiamata.
        loader:    DataLoader del training set di PHEME.
        criterion: BCEWithLogitsLoss.
        device:    Dispositivo di calcolo.

    Returns:
        Tupla (fisher_dict, opt_params_dict):
          fisher_dict:     {nome_param: tensore FIM diagonale}
          opt_params_dict: {nome_param: clone pesi ottimali theta*}
    """
    fisher_dict:     dict[str, torch.Tensor] = {}
    opt_params_dict: dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            opt_params_dict[name] = param.data.clone()
            fisher_dict[name]     = torch.zeros_like(param.data)

    model.eval()
    n_batches = len(loader)

    for batch_data in loader:
        batch_data = batch_data.to(device)
        model.zero_grad()

        out  = model(batch_data)
        loss = criterion(out.squeeze(), batch_data.y.squeeze())
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Accumulazione normalizzata: media su tutti i batch
                fisher_dict[name] += (
                    param.grad.data.pow(2) / n_batches
                )

    n_protected = len(fisher_dict)
    print(
        f"Fisher Information Matrix calcolata: "
        f"{n_protected} tensori protetti."
    )
    return fisher_dict, opt_params_dict


def compute_ewc_penalty(
    model:           HybridGatedBiGCN,
    fisher_dict:     dict[str, torch.Tensor],
    opt_params_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Calcola la penale EWC come somma pesata degli scostamenti da theta*.

    Implementa il termine di regolarizzazione della formula EWC
    (Kirkpatrick et al., 2017), escluso il coefficiente lambda/2
    che viene applicato esternamente nel loop di training:

        penalty = sum_i [ F_i * (theta_i - theta*_i)^2 ]

    La separazione dal coefficiente lambda consente di modificare
    LAMBDA_EWC senza ricalcolare la penale e di loggare separatamente
    il contributo della loss_task e della loss_ewc per diagnosi.

    Args:
        model:           Modello in fase di fine-tuning su USE24.
        fisher_dict:     FIM diagonale calcolata su PHEME.
        opt_params_dict: Pesi ottimali theta* post-training PHEME.

    Returns:
        Tensore scalare Float32 con la penale non scalata.
    """
    penalty = torch.tensor(
        0.0, device=next(model.parameters()).device
    )
    for name, param in model.named_parameters():
        if param.requires_grad and name in fisher_dict:
            f = fisher_dict[name]
            p = opt_params_dict[name]
            penalty += (f * (param - p).pow(2)).sum()
    return penalty


# ##############################################################################
# ### FASE 4: LOOP DI TRAINING E VALUTAZIONE                                ###
# ##############################################################################

def train_one_epoch_standard(
    model:     HybridGatedBiGCN,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device:    torch.device,
    clip_norm: float = CLIP_NORM,
) -> float:
    """Esegue una singola epoca di training standard senza penale EWC.

    Condivisa tra il training principale su PHEME e il fine-tuning
    Naive su USE24 (Ablation Study): la differenza sta nel modello
    e nel DataLoader passati, non nella logica di aggiornamento.

    Il gradient clipping (max_norm=1.0) e' applicato sistematicamente:
    le convoluzioni GCN su grafi con branching factor elevato (PHEME)
    producono gradienti con varianza alta nelle prime epoche.

    Args:
        model:     HybridGatedBiGCN in modalita' train.
        loader:    DataLoader PyG del set di addestramento.
        criterion: BCEWithLogitsLoss.
        optimizer: AdamW.
        device:    Dispositivo di calcolo.
        clip_norm: Soglia norma L2 per gradient clipping.

    Returns:
        Loss media per mini-batch sull'intera epoca.
    """
    model.train()
    total_loss = 0.0

    for batch_data in loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()

        out  = model(batch_data)
        loss = criterion(out.squeeze(), batch_data.y.squeeze())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_norm
        )
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_one_epoch_ewc(
    model:           HybridGatedBiGCN,
    loader:          DataLoader,
    criterion:       nn.Module,
    optimizer:       optim.Optimizer,
    device:          torch.device,
    fisher_dict:     dict[str, torch.Tensor],
    opt_params_dict: dict[str, torch.Tensor],
    lambda_ewc:      float = LAMBDA_EWC,
    clip_norm:       float = CLIP_NORM,
) -> tuple[float, float]:
    """Esegue una singola epoca di fine-tuning con penale EWC.

    La loss combinata e':
        L_tot = L_task + (lambda_ewc / 2) * penalty_EWC

    dove L_task e' la BCE sul task corrente (USE24) e penalty_EWC
    penalizza le deviazioni dai pesi theta* proporzionalmente alla
    loro importanza per il task storico (misurata dalla FIM).

    Restituisce sia la loss combinata sia la sola loss_task per
    consentire il monitoraggio del contributo relativo di EWC.

    Args:
        model:           HybridGatedBiGCN in modalita' train.
        loader:          DataLoader PyG training USE24.
        criterion:       BCEWithLogitsLoss.
        optimizer:       AdamW.
        device:          Dispositivo di calcolo.
        fisher_dict:     FIM diagonale calcolata su PHEME.
        opt_params_dict: Pesi theta* post-training PHEME.
        lambda_ewc:      Coefficiente di bilanciamento EWC.
        clip_norm:       Soglia gradient clipping.

    Returns:
        Tupla (loss_combinata_media, loss_task_media) per mini-batch.
    """
    model.train()
    total_combined = 0.0
    total_task     = 0.0

    for batch_data in loader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()

        out       = model(batch_data)
        loss_task = criterion(out.squeeze(), batch_data.y.squeeze())
        loss_ewc  = compute_ewc_penalty(
            model, fisher_dict, opt_params_dict
        )
        loss      = loss_task + (lambda_ewc / 2.0) * loss_ewc

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=clip_norm
        )
        optimizer.step()

        total_combined += loss.item()
        total_task     += loss_task.item()

    n = len(loader)
    return total_combined / n, total_task / n


def evaluate_graph_loader(
    model:  HybridGatedBiGCN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Esegue l'inferenza su un DataLoader PyG con gestione batch singolo.

    Imposta eval() e disabilita il grafo computazionale tramite no_grad()
    per minimizzare il consumo di VRAM durante la valutazione.

    Gestione batch singolo: squeeze() su [1,1] produce uno scalare
    (dim=0). unsqueeze(0) lo ripristina a [1] per garantire
    compatibilita' con list.extend() e BCEWithLogitsLoss.

    Args:
        model:  HybridGatedBiGCN.
        loader: DataLoader PyG del set di valutazione o test.
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

            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            if y_true.dim() == 0:
                y_true = y_true.unsqueeze(0)

            total_loss += criterion(logits, y_true).item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_true.long().cpu().tolist())

    return total_loss / len(loader), all_preds, all_labels


def run_training(
    model:        HybridGatedBiGCN,
    train_loader: DataLoader,
    val_loader:   Optional[DataLoader],
    device:       torch.device,
    epochs:       int   = EPOCHS_MAIN,
    lr:           float = LR_MAIN,
    wd:           float = WD_MAIN,
) -> None:
    """Esegue il loop di training principale su PHEME.

    Usa AdamW (Loshchilov e Hutter, 2019) per il weight decay decoupled:
    la regolarizzazione L2 e' applicata direttamente ai pesi anziche'
    al gradiente accumulato, producendo una regolarizzazione uniforme
    sui parametri del gate, delle proiezioni e delle GCNConv.

    Args:
        model:        HybridGatedBiGCN da addestrare.
        train_loader: DataLoader training PHEME.
        val_loader:   DataLoader validation PHEME, oppure None.
        device:       Dispositivo di calcolo.
        epochs:       Numero di epoche.
        lr:           Learning rate AdamW.
        wd:           Weight decay AdamW.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd
    )

    for epoch in range(1, epochs + 1):
        t_loss = train_one_epoch_standard(
            model, train_loader, criterion, optimizer, device
        )
        if val_loader is not None:
            v_loss, _, _ = evaluate_graph_loader(
                model, val_loader, device
            )
            print(
                f"Epoca [{epoch:02d}/{epochs}]  "
                f"Loss Train: {t_loss:.4f}  "
                f"Loss Val: {v_loss:.4f}"
            )
        else:
            print(
                f"Epoca [{epoch:02d}/{epochs}]  "
                f"Loss Train: {t_loss:.4f}"
            )


def run_finetuning_naive(
    model:        HybridGatedBiGCN,
    train_loader: DataLoader,
    device:       torch.device,
    epochs:       int   = EPOCHS_FT,
    lr:           float = LR_FT,
    wd:           float = WD_FT,
) -> None:
    """Esegue il fine-tuning Naive su USE24 senza penale EWC.

    Usato nell'Ablation Study (Backward Transfer) per dimostrare la
    Dimenticanza Catastrofica: il modello clone si adatta a USE24
    senza alcun vincolo sulla memoria storica di PHEME, producendo
    un crollo delle prestazioni atteso sul test set 2016.

    Args:
        model:        Clone del modello storico (copy.deepcopy).
        train_loader: DataLoader training USE24.
        device:       Dispositivo di calcolo.
        epochs:       Numero di epoche di fine-tuning.
        lr:           Learning rate.
        wd:           Weight decay.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd
    )
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch_standard(
            model, train_loader, criterion, optimizer, device
        )
        print(
            f"Epoca Naive [{epoch:02d}/{epochs}]  "
            f"Loss Train: {loss:.4f}"
        )


def run_finetuning_ewc(
    model:           HybridGatedBiGCN,
    train_loader:    DataLoader,
    device:          torch.device,
    fisher_dict:     dict[str, torch.Tensor],
    opt_params_dict: dict[str, torch.Tensor],
    epochs:          int   = EPOCHS_FT,
    lr:              float = LR_FT,
    wd:              float = WD_FT,
    lambda_ewc:      float = LAMBDA_EWC,
) -> None:
    """Esegue il fine-tuning EWC su USE24 per il Continual Learning.

    Adatta il modello al task 2024 minimizzando simultaneamente la BCE
    sul nuovo dominio e la penale EWC sul dominio storico. Il logging
    di entrambe le componenti (loss_combinata, loss_task) consente di
    verificare che la penale non sovrasti il segnale di addestramento.

    Args:
        model:           HybridGatedBiGCN post-training PHEME.
        train_loader:    DataLoader training USE24.
        device:          Dispositivo di calcolo.
        fisher_dict:     FIM diagonale estratta su PHEME.
        opt_params_dict: Pesi theta* post-training PHEME.
        epochs:          Numero di epoche di fine-tuning.
        lr:              Learning rate.
        wd:              Weight decay.
        lambda_ewc:      Coefficiente di bilanciamento EWC.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd
    )
    for epoch in range(1, epochs + 1):
        l_comb, l_task = train_one_epoch_ewc(
            model, train_loader, criterion, optimizer, device,
            fisher_dict, opt_params_dict, lambda_ewc,
        )
        print(
            f"Epoca EWC [{epoch:02d}/{epochs}]  "
            f"Loss Combinata: {l_comb:.4f}  "
            f"Loss Task: {l_task:.4f}"
        )


def print_evaluation_report(
    all_labels:   list[int],
    all_preds:    list[int],
    dataset_name: str,
    class_names:  list[str] = CLASS_NAMES,
) -> None:
    """Stampa accuratezza e classification report scikit-learn.

    Args:
        all_labels:   Lista di label vere (interi 0/1).
        all_preds:    Lista di predizioni del modello (interi 0/1).
        dataset_name: Stringa descrittiva per l'intestazione del report.
        class_names:  Nomi delle classi per il report sklearn.
    """
    acc = accuracy_score(all_labels, all_preds)
    print(
        f"\nAccuratezza su {dataset_name}: "
        f"{acc:.4f} ({acc*100:.1f}%)"
    )
    print(f"\nClassification Report — {dataset_name}:")
    print(classification_report(
        all_labels, all_preds, target_names=class_names
    ))


# ##############################################################################
# ### FASE 5: ENTRY POINT — ORCHESTRAZIONE DEGLI ESPERIMENTI                ###
# ##############################################################################

if __name__ == "__main__":

    print(f"Dispositivo di calcolo: {DEVICE}")
    SEP = "=" * 60

    # ------------------------------------------------------------------ #
    # STEP 1: Training baseline ibrida su PHEME (2016)                  #
    # Split 70 / 15 / 15 — il test_loader di PHEME sara' riutilizzato  #
    # negli Step 5 e 6 per il Backward Transfer (Ablation Study).       #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  STEP 1: Training Baseline Ibrida — PHEME (2016)")
    print(SEP)

    print("\nCaricamento e join label — PHEME...")
    pheme_graphs, pheme_labels = load_and_map_graphs(
        parquet_path = IN_PHEME_PARQUET,
        pt_path      = IN_PHEME_GRAPHS,
        label_map    = PHEME_LABEL_MAP,
    )

    print("\nCreazione DataLoader PHEME (70 / 15 / 15)...")
    pheme_loaders = create_graph_loaders(
        graphs = pheme_graphs,
        labels = pheme_labels,
        splits = {"train": 0.7, "val": 0.15, "test": 0.15},
    )

    model_hybrid = HybridGatedBiGCN(
        in_feats     = IN_FEATS,
        hidden_feats = HIDDEN_FEATS,
        common_dim   = COMMON_DIM,
        dropout_rate = DROPOUT_RATE,
    ).to(DEVICE)

    print("\nAvvio training Gated Bi-GCN su PHEME...")
    run_training(
        model        = model_hybrid,
        train_loader = pheme_loaders["train"],
        val_loader   = pheme_loaders["val"],
        device       = DEVICE,
    )

    print("\nValutazione sul test set di PHEME (baseline ibrida)...")
    _, preds_ph_base, labels_ph_base = evaluate_graph_loader(
        model_hybrid, pheme_loaders["test"], DEVICE
    )
    print_evaluation_report(
        labels_ph_base, preds_ph_base,
        "PHEME Test Set — Baseline Ibrida (2016)"
    )

    # ------------------------------------------------------------------ #
    # STEP 2: Clone del modello storico per l'Ablation Study            #
    # Il deepcopy isola una copia esatta dei pesi ottimali post-PHEME.  #
    # Questo clone sara' addestrato senza EWC nello Step 5 per          #
    # simulare la Dimenticanza Catastrofica da fine-tuning naive.       #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  STEP 2: Clone del Modello Storico (model_naive)")
    print(SEP)

    model_naive = copy.deepcopy(model_hybrid)
    print(
        "Clone isolato salvato come model_naive. "
        "I pesi riflettono theta* post-training PHEME."
    )

    # ------------------------------------------------------------------ #
    # STEP 3: Caricamento USE24 e test a freddo (Concept Drift)         #
    # Il campione e' il 20% stratificato dell'intero dataset USE24.     #
    # Il modello non riceve alcun aggiornamento dei pesi: questo e' un  #
    # test di inferenza pura per misurare il Concept Drift 2016->2024.  #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  STEP 3: Test a Freddo su USE24 (Concept Drift)")
    print(SEP)

    print("\nCaricamento e join label — USE24...")
    use24_graphs, use24_labels = load_and_map_graphs(
        parquet_path = IN_USE24_PARQUET,
        pt_path      = IN_USE24_GRAPHS,
        label_map    = USE24_LABEL_MAP,
    )

    print("\nCampionamento 20% stratificato per il test a freddo...")
    cold_loaders = create_graph_loaders(
        graphs      = use24_graphs,
        labels      = use24_labels,
        splits      = {"test": 1.0},
        sample_frac = 0.20,
    )

    print(
        "\nValutazione a freddo — "
        "model_hybrid (PHEME 2016) su USE24 (2024)..."
    )
    _, preds_cold, labels_cold = evaluate_graph_loader(
        model_hybrid, cold_loaders["test"], DEVICE
    )
    print_evaluation_report(
        labels_cold, preds_cold,
        "USE24 — Concept Drift a freddo (2016 -> 2024)"
    )

    # ------------------------------------------------------------------ #
    # STEP 4: Estrazione Fisher Information Matrix su PHEME              #
    # La FIM viene calcolata sul training set di PHEME usando i pesi     #
    # ottimali theta* del modello post-training (prima di qualsiasi     #
    # fine-tuning). Fisher_dict e opt_params_dict vengono passati alla  #
    # penale EWC nello Step 6 per proteggere la memoria storica.        #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  STEP 4: Estrazione Fisher Information Matrix (PHEME)")
    print(SEP)

    criterion_fisher = nn.BCEWithLogitsLoss()
    fisher_dict, opt_params_dict = compute_fisher_matrix(
        model     = model_hybrid,
        loader    = pheme_loaders["train"],
        criterion = criterion_fisher,
        device    = DEVICE,
    )

    # ------------------------------------------------------------------ #
    # STEP 5 (prep): Split USE24 per il fine-tuning (80 / 20)           #
    # Il test_loader_use24 e' separato dal cold_loader dello Step 3:    #
    # quest'ultimo era un campione del 20%; qui usiamo l'intero dataset  #
    # suddiviso in 80% train e 20% test per una valutazione equa del    #
    # fine-tuning sia Naive che EWC.                                    #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  STEP 5: Split USE24 per Fine-Tuning (80 / 20)")
    print(SEP)

    use24_ft_loaders = create_graph_loaders(
        graphs = use24_graphs,
        labels = use24_labels,
        splits = {"train": 0.8, "test": 0.2},
    )

    # ------------------------------------------------------------------ #
    # STEP 5: Fine-tuning Naive del clone storico su USE24              #
    # model_naive viene addestrato sullo stesso train_loader di USE24    #
    # usato per l'EWC, ma senza alcun vincolo sulla memoria storica.    #
    # La degradazione attesa su PHEME (Step 7a) costituira' la          #
    # misura quantitativa della Dimenticanza Catastrofica.              #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("  STEP 5: Fine-Tuning Naive su USE24 (senza EWC)")
    print(SEP)

    print(
        f"\nAvvio fine-tuning Naive su USE24 "
        f"({EPOCHS_FT} epoche, lr={LR_FT})..."
    )
    run_finetuning_naive(
        model        = model_naive,
        train_loader = use24_ft_loaders["train"],
        device       = DEVICE,
    )

    # ------------------------------------------------------------------ #
    # STEP 6: Fine-tuning EWC del modello principale su USE24           #
    # model_hybrid e' aggiornato sul task 2024 con la penale EWC        #
    # (lambda=5000) che protegge i pesi importanti per PHEME.           #
    # Il test sul use24_ft_loaders["test"] misura l'adattamento al      #
    # task corrente; il test su PHEME nello Step 7b misura la           #
    # retention della memoria storica.                                  #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print(
        f"  STEP 6: Fine-Tuning EWC su USE24 "
        f"(lambda={LAMBDA_EWC})"
    )
    print(SEP)

    print(
        f"\nAvvio fine-tuning EWC su USE24 "
        f"({EPOCHS_FT} epoche, lr={LR_FT})..."
    )
    run_finetuning_ewc(
        model           = model_hybrid,
        train_loader    = use24_ft_loaders["train"],
        device          = DEVICE,
        fisher_dict     = fisher_dict,
        opt_params_dict = opt_params_dict,
    )

    print("\nValutazione post-EWC sul test set di USE24...")
    _, preds_ewc_u24, labels_ewc_u24 = evaluate_graph_loader(
        model_hybrid, use24_ft_loaders["test"], DEVICE
    )
    print_evaluation_report(
        labels_ewc_u24, preds_ewc_u24,
        "USE24 — Post Fine-Tuning EWC (2024)"
    )

    # ------------------------------------------------------------------ #
    # STEP 7: Ablation Study — Backward Transfer su PHEME               #
    # Entrambi i modelli (Naive e EWC) vengono valutati sul test set    #
    # originale di PHEME (pheme_loaders["test"]), che non e' stato      #
    # toccato durante il fine-tuning. Il delta di accuratezza tra       #
    # model_naive e model_hybrid su PHEME quantifica la quota di        #
    # memoria storica preservata dalla regolarizzazione EWC.            #
    #                                                                    #
    # Risultato atteso:                                                  #
    #   model_naive -> crollo su PHEME (Dimenticanza Catastrofica)       #
    #   model_hybrid -> degradazione contenuta (Memoria Preservata)     #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print(
        "  STEP 7: Ablation Study — Backward Transfer su PHEME"
    )
    print(SEP)

    print(
        "\n7a — model_naive (Naive Fine-Tuning) "
        "su PHEME 2016..."
    )
    _, preds_naive_ph, labels_naive_ph = evaluate_graph_loader(
        model_naive, pheme_loaders["test"], DEVICE
    )
    print_evaluation_report(
        labels_naive_ph, preds_naive_ph,
        "PHEME — Backward Transfer Naive "
        "(Dimenticanza Catastrofica)"
    )

    print(
        "\n7b — model_hybrid (EWC Fine-Tuning) "
        "su PHEME 2016..."
    )
    _, preds_ewc_ph, labels_ewc_ph = evaluate_graph_loader(
        model_hybrid, pheme_loaders["test"], DEVICE
    )
    print_evaluation_report(
        labels_ewc_ph, preds_ewc_ph,
        "PHEME — Backward Transfer EWC "
        "(Memoria Preservata)"
    )

    print(f"\n{SEP}")
    print("  Pipeline Modello Ibrido + EWC completata.")
    print(SEP)
