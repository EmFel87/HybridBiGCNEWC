# =============================================================================
# FILE:    models/semantic.py
# SCOPO:   Definizione delle classi architetturali per la baseline
#          semantica del progetto "Fake News Detection: 2016 vs 2024".
#          Contiene esclusivamente il Dataset PyTorch che decodifica gli
#          embedding BERTweet serializzati e il classificatore
#          Feed-Forward che opera su tali embedding.
#          Nessuna logica di training, valutazione o gestione dei seed
#          risiede in questo modulo: e' responsabilita' esclusiva di
#          run_semantic.py, che importa queste classi.
# DIPENDENZE: torch, numpy, pandas
# MODULO:  Architetture — Baseline Semantica
# =============================================================================

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TextEmbeddingDataset(Dataset):
    """Dataset PyTorch per embedding testuali serializzati in Float16.

    Legge la colonna ``embedding_bin`` di un DataFrame Pandas, decodifica
    il buffer di byte in un vettore Float32 di 768 dimensioni (l'output
    del token [CLS] di BERTweet) e lo restituisce come tensore PyTorch
    insieme alla label binaria.

    La deserializzazione da Float16 a Float32 avviene in ``__getitem__``
    anziche' nel costruttore, per mantenere basso il consumo di RAM
    durante il caricamento (decodifica lazy, a runtime, per singolo
    campione).

    Attributes:
        df: DataFrame Pandas con indice resettato, contenente le
            colonne ``embedding_bin`` e ``target_col``.
        target_col: Nome della colonna con le label numeriche intere.
    """

    def __init__(
        self,
        dataframe:  pd.DataFrame,
        target_col: str = "label_num",
    ) -> None:
        """Inizializza il dataset a partire da un DataFrame Pandas.

        Args:
            dataframe: DataFrame contenente almeno le colonne
                ``embedding_bin`` (bytes, embedding Float16 serializzato)
                e ``target_col`` (label binaria intera).
            target_col: Nome della colonna con le label numeriche.
                Default: ``"label_num"``.
        """
        self.df         = dataframe.reset_index(drop=True)
        self.target_col = target_col

    def __len__(self) -> int:
        """Restituisce il numero di campioni nel dataset.

        Returns:
            Numero di righe del DataFrame sottostante.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Restituisce il vettore di embedding e la label del campione idx.

        Args:
            idx: Indice del campione nel DataFrame.

        Returns:
            Tupla ``(x, y)`` dove:
              ``x``: tensore Float32 di forma ``[768]``.
              ``y``: tensore Float32 scalare (richiesto da
                ``BCEWithLogitsLoss``).
        """
        row     = self.df.iloc[idx]
        emb_raw = row["embedding_bin"]

        vec = np.frombuffer(emb_raw, dtype=np.float16).astype(np.float32)
        x   = torch.tensor(vec)

        y = torch.tensor(int(row[self.target_col]), dtype=torch.float32)
        return x, y


class SemanticBaselineModel(torch.nn.Module):
    """Classificatore binario Feed-Forward su embedding BERTweet [CLS].

    Architettura a tre layer lineari con Dropout, progettata per operare
    sull'embedding aggregato [CLS] prodotto da BERTweet (768 dimensioni).
    Non accede alla sequenza completa di token: costituisce la baseline
    "solo testo" del confronto con l'architettura topologica Bi-GCN.

    Struttura:
        ``Linear(768 -> 256) -> ReLU -> Dropout(0.3)``
        ``Linear(256 ->  64) -> ReLU -> Dropout(0.3)``
        ``Linear( 64 ->   1)``  [logit grezzo]

    L'ultimo strato produce un singolo logit non normalizzato. La
    conversione in probabilita' tramite sigmoide e' responsabilita'
    del chiamante: ``BCEWithLogitsLoss`` durante il training (per
    stabilita' numerica) e ``torch.sigmoid()`` durante l'inferenza.
    """

    def __init__(
        self,
        input_dim:    int   = 768,
        hidden_dim:   int   = 256,
        dropout_rate: float = 0.3,
    ) -> None:
        """Inizializza gli strati lineari del classificatore.

        Args:
            input_dim: Dimensione del vettore di input. Default: 768
                (dimensione dell'embedding [CLS] di BERTweet).
            hidden_dim: Dimensione del primo layer nascosto.
                Default: 256.
            dropout_rate: Probabilita' di dropout applicata dopo ogni
                blocco Linear+ReLU. Default: 0.3.
        """
        super(SemanticBaselineModel, self).__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),

            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),

            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcola il logit grezzo per un batch di embedding.

        Args:
            x: Tensore Float32 di forma ``[B, input_dim]``.

        Returns:
            Tensore Float32 di forma ``[B, 1]`` contenente i logit non
            normalizzati. Applicare ``torch.sigmoid()`` per ottenere le
            probabilita' di classe Real (1).
        """
        return self.network(x)
