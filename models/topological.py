# =============================================================================
# FILE:    models/topological.py
# SCOPO:   Definizione dell'architettura per la baseline topologica del
#          progetto "Fake News Detection: 2016 vs 2024". Contiene
#          esclusivamente la rete Bidirectional Graph Convolutional
#          Network (Bi-GCN) che opera sulle cascate di propagazione
#          rappresentate come oggetti torch_geometric.data.Data.
#
#          Nota sul dataset: il notebook originale non definisce una
#          classe Dataset personalizzata per i grafi. I file .pt
#          prodotti dalla feature extraction topologica contengono gia'
#          liste di oggetti torch_geometric.data.Data pronti all'uso,
#          che vengono passati direttamente a torch_geometric.loader.
#          DataLoader senza un wrapper Dataset intermedio (a differenza
#          della baseline semantica, che richiede TextEmbeddingDataset
#          per decodificare gli embedding Float16 on-the-fly). Per
#          coerenza con il codice originale, questo modulo non introduce
#          una classe Dataset che non era presente nella pipeline.
#
#          Nessuna logica di training, valutazione o gestione dei seed
#          risiede in questo modulo: e' responsabilita' esclusiva di
#          run_topological.py, che importa questa classe.
# DIPENDENZE: torch, torch_geometric
# MODULO:  Architetture — Baseline Topologica (Bi-GCN)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class BiGCNBaselineModel(nn.Module):
    """Rete Bi-GCN per la classificazione binaria di cascate di propagazione.

    Implementa l'architettura Bidirectional Graph Convolutional Network
    (Bian et al., 2020), che processa ogni cascata di propagazione
    attraverso due rami di convoluzione grafica paralleli e indipendenti:

    Ramo Top-Down (TD):
        Modella il flusso dell'informazione dalla radice verso le foglie,
        seguendo l'orientamento nativo di ``edge_index`` (parent -> child).
        Cattura il pattern di diffusione della notizia originale.

    Ramo Bottom-Up (BU):
        Modella il flusso di feedback dalle foglie verso la radice,
        ottenuto invertendo l'orientamento degli archi tramite
        ``edge_index.flip(0)`` (child -> parent). Cattura la risposta
        aggregata dell'audience alla notizia.

    I vettori aggregati dei due rami (ottenuti tramite
    ``global_mean_pool``) vengono concatenati e proiettati da un
    classificatore Feed-Forward a due strati verso un singolo logit
    binario.

    Struttura:
        ``[TD]  GCNConv(in_feats->hidden) -> ReLU -> GCNConv(hidden->hidden)
               -> ReLU -> MeanPool``
        ``[BU]  GCNConv(in_feats->hidden) -> ReLU -> GCNConv(hidden->hidden)
               -> ReLU -> MeanPool``  [su edge_index invertito]
        ``[FC]  Linear(hidden*2->hidden) -> ReLU -> Dropout(0.3)
               -> Linear(hidden->1)``
    """

    def __init__(
        self,
        in_feats:     int = 768,
        hidden_feats: int = 64,
    ) -> None:
        """Inizializza le convoluzioni grafiche e il classificatore finale.

        Args:
            in_feats: Dimensione delle feature per nodo. Default: 768
                (dimensione dell'embedding [CLS] di BERTweet associato
                a ciascun nodo della cascata).
            hidden_feats: Dimensione degli hidden state prodotti da
                ciascuna convoluzione GCN. Default: 64.
        """
        super(BiGCNBaselineModel, self).__init__()

        # -- Ramo Top-Down -----------------------------------------------------
        self.td_conv1 = GCNConv(in_feats, hidden_feats)
        self.td_conv2 = GCNConv(hidden_feats, hidden_feats)

        # -- Ramo Bottom-Up ------------------------------------------------------
        self.bu_conv1 = GCNConv(in_feats, hidden_feats)
        self.bu_conv2 = GCNConv(hidden_feats, hidden_feats)

        # -- Classificatore finale -----------------------------------------------
        # Input: concatenazione TD || BU -> hidden_feats * 2
        self.fc1     = nn.Linear(hidden_feats * 2, hidden_feats)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(hidden_feats, 1)

    def forward(self, data: "torch_geometric.data.Batch") -> torch.Tensor:
        """Calcola il logit binario per un batch di grafi.

        Args:
            data: Batch PyTorch Geometric (oggetto ``Batch``) contenente
                gli attributi:
                  ``x``          — feature matrix dei nodi
                      ``[N_tot, in_feats]``.
                  ``edge_index`` — archi orientati top-down
                      ``[2, E_tot]``.
                  ``batch``      — vettore di assegnazione
                      nodo -> grafo ``[N_tot]``.

        Returns:
            Tensore Float32 di forma ``[B, 1]`` con i logit non
            normalizzati. Applicare ``torch.sigmoid()`` per ottenere le
            probabilita' di classe Real (1).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Inversione degli archi per il ramo Bottom-Up: flip(0) scambia
        # le righe [src, dst] -> [dst, src], equivalente alla trasposta
        # della matrice di adiacenza.
        edge_index_bu = edge_index.flip(0)

        # -- Propagazione Top-Down -------------------------------------------
        x_td = F.relu(self.td_conv1(x, edge_index))
        x_td = F.relu(self.td_conv2(x_td, edge_index))
        g_td = global_mean_pool(x_td, batch)

        # -- Propagazione Bottom-Up ------------------------------------------
        x_bu = F.relu(self.bu_conv1(x, edge_index_bu))
        x_bu = F.relu(self.bu_conv2(x_bu, edge_index_bu))
        g_bu = global_mean_pool(x_bu, batch)

        # -- Fusione e classificazione ----------------------------------------
        g_combined = torch.cat([g_td, g_bu], dim=1)

        out = F.relu(self.fc1(g_combined))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
