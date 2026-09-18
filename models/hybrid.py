# =============================================================================
# FILE:    models/hybrid.py
# SCOPO:   Definizione dell'architettura ibrida Gated Bi-GCN del progetto
#          "Fake News Detection: 2016 vs 2024". Contiene esclusivamente
#          la classe HybridGatedBiGCN, che fonde un ramo topologico
#          (Bi-GCN su archi Top-Down/Bottom-Up) con un ramo semantico
#          (proiezione dell'embedding BERTweet del nodo radice) tramite
#          un meccanismo di gating adattivo a sigmoide.
#          Nessuna logica di training, valutazione, gestione dei seed o
#          Continual Learning (EWC) risiede in questo modulo: la
#          regolarizzazione EWC vive in models/ewc.py, l'orchestrazione
#          del training in run_hybrid.py.
# DIPENDENZE: torch, torch_geometric
# MODULO:  Architetture — Modello Ibrido (Gated Bi-GCN)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class HybridGatedBiGCN(nn.Module):
    """Classificatore ibrido Gated Bi-GCN per cascate di propagazione.

    Fonde un ramo topologico Bi-GCN con un ramo semantico BERTweet
    tramite un modulo di gating adattivo a sigmoide, per componente.
    L'architettura risolve il disallineamento strutturale tra PHEME
    (alberi gerarchici, segnale topologico forte) e USE24 (nodi isolati,
    segnale topologico nullo): il gate apprende autonomamente a pesare
    i due contributi in base alla ricchezza della struttura del grafo.

    Comportamento atteso del gate (parametro ``alpha``):
      ``alpha -> 1`` per USE24: nodi isolati, il ramo testuale domina.
      ``alpha -> 0`` per PHEME: alberi profondi, la topologia domina.

    Struttura interna:
      ``[TD]``   ``GCNConv(in_feats->hidden,GELU) x2 -> MeanPool``
      ``[BU]``   ``GCNConv(in_feats->hidden,GELU) x2 -> MeanPool``
                 (su ``edge_index`` invertito tramite ``flip(0)``)
      ``[Proj_topo]``  ``Linear(hidden*2->common_dim) -> LayerNorm
                 -> GELU``
      ``[Proj_sem]``   ``Linear(in_feats->common_dim) -> LayerNorm
                 -> GELU``
      ``[Gate]``       ``Linear(common_dim*2->common_dim) -> Sigmoid``
      ``[Fusion]``     ``alpha * z_sem + (1 - alpha) * z_topo``
      ``[FC]``         ``Linear(common_dim->64,GELU) -> Dropout(0.3)
                 -> Linear(64->1)``

    GELU e' usato sistematicamente al posto di ReLU (incluso il
    classificatore finale) per mantenere gradienti non nulli sulle
    attivazioni leggermente negative: proprieta' rilevante per il
    calcolo accurato della Fisher Information Matrix durante la fase
    di Continual Learning (vedi ``models/ewc.py``).

    Attributes:
        last_alpha: Tensore ``[B, common_dim]`` contenente l'ultimo
            vettore di gating calcolato durante l'ultima chiamata a
            ``forward``, salvato con ``.detach()`` per l'ispezione
            post-hoc del comportamento del gate (es. media di
            ``alpha`` su un dataset per confrontare quanto il modello
            si affida al testo vs alla topologia). Non fa parte del
            grafo computazionale e non deve essere usato per il
            backward pass.
    """

    def __init__(
        self,
        in_feats:     int = 768,
        hidden_feats: int = 64,
        common_dim:   int = 128,
    ) -> None:
        """Inizializza le convoluzioni grafiche, le proiezioni e il gate.

        Args:
            in_feats: Dimensione delle feature per nodo, prodotte da
                BERTweet. Default: 768.
            hidden_feats: Dimensione degli hidden state delle
                convoluzioni GCN (rami Top-Down e Bottom-Up).
                Default: 64.
            common_dim: Dimensione dello spazio comune in cui vengono
                proiettati sia il vettore topologico che quello
                semantico prima della fusione gated. Default: 128.
        """
        super(HybridGatedBiGCN, self).__init__()

        # -- Ramo topologico Top-Down -----------------------------------------
        self.td_conv1 = GCNConv(in_feats, hidden_feats)
        self.td_conv2 = GCNConv(hidden_feats, hidden_feats)

        # -- Ramo topologico Bottom-Up -----------------------------------------
        self.bu_conv1 = GCNConv(in_feats, hidden_feats)
        self.bu_conv2 = GCNConv(hidden_feats, hidden_feats)

        # -- Proiezione topologica: hidden_feats*2 -> common_dim ---------------
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

        # -- Modulo di fusione adattiva (Gate) ------------------------------------
        # Riceve z_sem || z_topo (common_dim * 2) e produce alpha in (0,1)
        # con granularita' per componente dello spazio comune.
        self.gate = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
            nn.Sigmoid(),
        )

        # -- Classificatore finale --------------------------------------------
        self.fc1     = nn.Linear(common_dim, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(64, 1)

        # Placeholder per l'ultimo vettore di gating calcolato in forward().
        # Popolato al primo forward pass; utile per l'analisi del
        # comportamento del gate senza dover ricalcolare manualmente
        # le proiezioni al di fuori del modello.
        self.last_alpha: torch.Tensor | None = None

    def forward(self, data: "torch_geometric.data.Batch", force_alpha: float | None = None) -> torch.Tensor:
        """Calcola il logit binario per un batch di grafi.

        Args:
            data: Batch PyTorch Geometric (oggetto ``Batch``) con
                attributi:
                  ``x``          — feature matrix nodi
                      ``[N_tot, in_feats]``.
                  ``edge_index`` — archi top-down ``[2, E_tot]``.
                  ``batch``      — assegnazione nodo->grafo
                      ``[N_tot]``.
                  ``ptr``        — indici di inizio per ogni grafo
                      nel batch, usati per estrarre il nodo radice
                      di ciascuna cascata.

        Returns:
            Tensore Float32 ``[B, 1]`` con logit non normalizzati.
            Applicare ``torch.sigmoid()`` per le probabilita' di
            classe.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Inversione degli archi per il ramo Bottom-Up: flip(0) scambia
        # le righe [src, dst] -> [dst, src], equivalente alla trasposta
        # della matrice di adiacenza.
        edge_index_bu = edge_index.flip(0)

        # -- Propagazione Top-Down ---------------------------------------------
        x_td = F.gelu(self.td_conv1(x, edge_index))
        x_td = F.gelu(self.td_conv2(x_td, edge_index))
        g_td = global_mean_pool(x_td, batch)  # [B, hidden_feats]

        # -- Propagazione Bottom-Up ---------------------------------------------
        x_bu = F.gelu(self.bu_conv1(x, edge_index_bu))
        x_bu = F.gelu(self.bu_conv2(x_bu, edge_index_bu))
        g_bu = global_mean_pool(x_bu, batch)  # [B, hidden_feats]

        # -- Proiezione topologica ------------------------------------------------
        z_topo = self.graph_proj(
            torch.cat([g_td, g_bu], dim=1)
        )  # [B, common_dim]

        # -- Estrazione embedding nodo radice (ramo semantico) --------------------
        # data.ptr[:-1] contiene l'indice del primo nodo di ogni grafo
        # nel super-grafo batch, corrispondente al tweet radice della
        # cascata (o all'unico nodo, nel caso di grafi USE24 isolati).
        root_indices = data.ptr[:-1]
        z_sem = self.text_proj(x[root_indices])  # [B, common_dim]

        # -- Fusione gated dinamica ------------------------------------------------
        alpha = self.gate(
            torch.cat([z_sem, z_topo], dim=1)
        )  # [B, common_dim]

        if force_alpha is not None:
            alpha = torch.full_like(alpha, force_alpha)

        # Salvataggio dell'ultimo alpha calcolato, isolato dal grafo
        # computazionale tramite detach(): consente l'ispezione del
        # comportamento del gate (es. model.last_alpha.mean()) senza
        # interferire con il backward pass del training corrente.
        self.last_alpha = alpha.detach()

        # Combinazione convessa per componente dello spazio comune:
        # se alpha -> 1 il modello preserva il testo, se alpha -> 0
        # il modello si affida al grafo.
        h_hybrid = alpha * z_sem + (1.0 - alpha) * z_topo  # [B, common_dim]

        # -- Classificazione ------------------------------------------------------
        out = self.fc2(self.dropout(F.gelu(self.fc1(h_hybrid))))
        return out
