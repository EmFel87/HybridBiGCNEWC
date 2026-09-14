# =============================================================================
# FILE:    models/ablation.py
# SCOPO:   Definizione dell'architettura di ablation study del progetto
#          "Fake News Detection: 2016 vs 2024". Contiene esclusivamente
#          la classe HybridGatedMLP: una variante di HybridGatedBiGCN in
#          cui le convoluzioni grafiche (GCNConv) del ramo topologico
#          sono sostituite da trasformazioni lineari indipendenti per
#          nodo (nn.Linear), che IGNORANO la struttura degli archi.
#          Questo modello serve a isolare empiricamente il contributo
#          del message passing sul grafo: qualunque differenza di
#          prestazioni rispetto a HybridGatedBiGCN e' attribuibile
#          esclusivamente alla capacita' di sfruttare la topologia
#          della cascata, non alla capacita' rappresentazionale delle
#          proiezioni o al meccanismo di gating (che restano identici).
#          Nessuna logica di training, valutazione, gestione dei seed o
#          Continual Learning (EWC) risiede in questo modulo.
# DIPENDENZE: torch, torch_geometric
# MODULO:  Architetture — Ablation Study (Gated MLP, senza message passing)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class HybridGatedMLP(nn.Module):
    """Variante di ablation di HybridGatedBiGCN priva di message passing.

    Condivide con ``HybridGatedBiGCN`` (in ``models/hybrid.py``) la
    stessa architettura di fusione — proiezioni testo/grafo, gate
    adattivo a sigmoide, classificatore finale — ma sostituisce le
    convoluzioni grafiche ``GCNConv`` con strati ``nn.Linear``
    applicati indipendentemente a ciascun nodo. Il risultato e' che il
    "ramo topologico" di questo modello elabora le feature dei nodi
    senza mai leggere ``edge_index``: ogni nodo viene trasformato in
    isolamento e poi aggregato con ``global_mean_pool`` esattamente
    come nel modello completo.

    Questo isola l'effetto del message passing: la differenza di
    prestazioni tra ``HybridGatedBiGCN`` e ``HybridGatedMLP`` a parita'
    di tutto il resto (proiezioni, gate, classificatore, regime di
    training) e' imputabile unicamente alla capacita' delle GCNConv di
    propagare informazione lungo la struttura della cascata.

    Struttura:
      ``[TD]``   ``Linear(in_feats->hidden,GELU) x2 -> MeanPool``
      ``[BU]``   ``Linear(in_feats->hidden,GELU) x2 -> MeanPool``
                 (applicato agli stessi nodi, senza edge_index)
      ``[Proj_topo]``  ``Linear(hidden*2->common_dim) -> LayerNorm
                 -> GELU``
      ``[Proj_sem]``   ``Linear(in_feats->common_dim) -> LayerNorm
                 -> GELU``
      ``[Gate]``       ``Linear(common_dim*2->common_dim) -> Sigmoid``
      ``[Fusion]``     ``alpha * z_sem + (1 - alpha) * z_topo``
      ``[FC]``         ``Linear(common_dim->64,GELU) -> Dropout(0.3)
                 -> Linear(64->1)``
    """

    def __init__(
        self,
        in_feats:     int = 768,
        hidden_feats: int = 64,
        common_dim:   int = 128,
    ) -> None:
        """Inizializza le trasformazioni lineari, le proiezioni e il gate.

        Args:
            in_feats: Dimensione delle feature per nodo, prodotte da
                BERTweet. Default: 768.
            hidden_feats: Dimensione dello spazio intermedio prodotto
                dalle trasformazioni lineari per nodo (equivalente
                strutturale dell'hidden state GCN). Default: 64.
            common_dim: Dimensione dello spazio comune in cui vengono
                proiettati sia il vettore "topologico" (in realta'
                puramente per-nodo) che quello semantico prima della
                fusione gated. Default: 128.
        """
        super(HybridGatedMLP, self).__init__()

        # -- Ramo "Top-Down": trasformazione lineare per nodo, nessun ---------
        # -- accesso a edge_index.
        self.td_lin1 = nn.Linear(in_feats, hidden_feats)
        self.td_lin2 = nn.Linear(hidden_feats, hidden_feats)

        # -- Ramo "Bottom-Up": trasformazione lineare indipendente -------------
        # -- (stessi nodi in input del ramo TD, pesi separati).
        self.bu_lin1 = nn.Linear(in_feats, hidden_feats)
        self.bu_lin2 = nn.Linear(hidden_feats, hidden_feats)

        # -- Proiezione "topologica": hidden_feats*2 -> common_dim -------------
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_feats * 2, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
        )

        # -- Proiezione semantica: in_feats -> common_dim -----------------------
        self.text_proj = nn.Sequential(
            nn.Linear(in_feats, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
        )

        # -- Modulo di fusione adattiva (Gate), identico a HybridGatedBiGCN ----
        self.gate = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
            nn.Sigmoid(),
        )

        # -- Classificatore finale, identico a HybridGatedBiGCN -----------------
        self.fc1     = nn.Linear(common_dim, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(64, 1)

    def forward(self, data: "torch_geometric.data.Batch") -> torch.Tensor:
        """Calcola il logit binario per un batch di grafi, ignorando gli archi.

        A differenza di ``HybridGatedBiGCN.forward``, questo metodo non
        legge mai ``data.edge_index``: le feature dei nodi vengono
        trasformate individualmente tramite ``nn.Linear`` e poi
        aggregate con ``global_mean_pool`` usando esclusivamente
        ``data.batch`` per l'assegnazione nodo -> grafo.

        Args:
            data: Batch PyTorch Geometric (oggetto ``Batch``) con
                attributi:
                  ``x``     — feature matrix nodi ``[N_tot, in_feats]``.
                  ``batch`` — assegnazione nodo -> grafo ``[N_tot]``.
                  ``ptr``   — indici di inizio per ogni grafo nel
                      batch, usati per estrarre il nodo radice.
                (``edge_index`` non viene utilizzato.)

        Returns:
            Tensore Float32 ``[B, 1]`` con logit non normalizzati.
            Applicare ``torch.sigmoid()`` per le probabilita' di
            classe.
        """
        x, batch = data.x, data.batch

        # -- Trasformazione "Top-Down" per nodo (nessun message passing) -------
        x_td = F.gelu(self.td_lin1(x))
        x_td = F.gelu(self.td_lin2(x_td))
        g_td = global_mean_pool(x_td, batch)  # [B, hidden_feats]

        # -- Trasformazione "Bottom-Up" per nodo (nessun message passing) ------
        x_bu = F.gelu(self.bu_lin1(x))
        x_bu = F.gelu(self.bu_lin2(x_bu))
        g_bu = global_mean_pool(x_bu, batch)  # [B, hidden_feats]

        # -- Proiezione "topologica" -----------------------------------------------
        z_topo = self.graph_proj(
            torch.cat([g_td, g_bu], dim=1)
        )  # [B, common_dim]

        # -- Estrazione embedding nodo radice (ramo semantico) ----------------------
        root_indices = data.ptr[:-1]
        z_sem = self.text_proj(x[root_indices])  # [B, common_dim]

        # -- Fusione gated dinamica -------------------------------------------------
        alpha = self.gate(
            torch.cat([z_sem, z_topo], dim=1)
        )  # [B, common_dim]
        h_hybrid = alpha * z_sem + (1.0 - alpha) * z_topo  # [B, common_dim]

        # -- Classificazione ---------------------------------------------------------
        out = self.fc2(self.dropout(F.gelu(self.fc1(h_hybrid))))
        return out
