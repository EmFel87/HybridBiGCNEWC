# =============================================================================
# FILE:    feature_extraction_topological.py
# SCOPO:   Conversione dei Parquet vettorizzati (output di
#          feature_extraction_semantic.py) in grafi PyTorch Geometric
#          (torch_geometric.data.Data) pronti per la rete Bi-GCN.
#          Gestisce due topologie eterogenee che costituiscono il nucleo
#          del Concept Drift rilevato tra PHEME (2016) e USE24 (2024):
#            - Albero gerarchico (PHEME): propagazione multi-livello
#              tramite parent_id esplicito nel dataset.
#            - Punti isolati/stella (USE24): ogni post è radice di sé
#              stesso, senza risposte annidate; il parent_id viene
#              sintetizzato dalla logica interna di questo modulo.
# DIPENDENZE: torch, torch_geometric, numpy, pandas, tqdm
# MODULO:  3 di 3 — da eseguire dopo feature_extraction_semantic.py
# =============================================================================


# ##############################################################################
# ### FASE 0: IMPORT E COSTANTI GLOBALI                                      ###
# ##############################################################################

import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm

# -- Percorsi Parquet di input (output di feature_extraction_semantic.py) -----
DRIVE_BASE      = "/content/drive/MyDrive/Tesi"
IN_PHEME        = f"{DRIVE_BASE}/Parquet_Finali/PHEME_Vectorized_FP16.parquet"
IN_USE24        = f"{DRIVE_BASE}/Parquet_Finali/USE24_Vectorized_FP16.parquet"

# -- Directory di output per i file .pt dei grafi ----------------------------
OUTPUT_DIR      = f"{DRIVE_BASE}/Grafi_PyG"

# -- Dimensione attesa del vettore di embedding BERTweet ----------------------
EMBED_DIM       = 768

# -- Mapping label testuale -> classe intera per classificazione --------------
#    CLASSE 0 — Fake/Rumor: include varianti ortografiche britanniche
#               (rumour, non-rumour) e tutte le categorie USE24-XD.
#    CLASSE 1 — True/Non-Rumor: include "neutral" per i post USE24-XD
#               classificati come privi di contenuto misinfomativo.
#    CLASSE 2 — Unverified: non confermato né smentito al momento
#               dell'annotazione; mantenuto come classe separata per
#               consentire esperimenti di classificazione ternaria.
LABEL_MAP: dict[str, int] = {
    # Classe 0: contenuto potenzialmente falso o fuorviante
    "rumor":           0,
    "rumour":          0,  # ortografia britannica (PHEME)
    "misinformation":  0,
    "sensationalism":  0,  # categoria USE24-XD
    "conspiracy":      0,  # categoria USE24-XD
    "hate_speech":     0,  # categoria USE24-XD
    "speculation":     0,  # categoria USE24-XD
    "satire":          0,  # categoria USE24-XD
    "debunked":        0,
    # Classe 1: contenuto verificato o neutro
    "non-rumor":       1,
    "non-rumour":      1,  # ortografia britannica (PHEME)
    "true":            1,
    "neutral":         1,  # nessuna categoria attiva in USE24-XD
    # Classe 2: stato di verifica indeterminato
    "unverified":      2,
}


# ##############################################################################
# ### FASE 1: COSTRUZIONE DEL SINGOLO GRAFO (create_bigcn_data)             ###
# ##############################################################################

def _inject_star_topology(df: pd.DataFrame) -> pd.DataFrame:
    """Sintetizza la colonna parent_id per dataset con topologia a stella.

    USE24-XD non contiene il campo parent_id perché ogni post e' un
    nodo radice autonomo: la struttura e' quella di un insieme di
    "punti isolati" (grafo disconnesso) piuttosto che di un albero
    gerarchico come in PHEME.

    Per consentire l'uso della stessa funzione create_bigcn_data su
    entrambi i dataset, questa funzione sintetizza un parent_id per
    i dataset che ne sono privi, creando una topologia a stella
    minima: tutti i nodi del gruppo puntano alla radice identificata
    da root_id, tranne la radice stessa (che non ha parent).

    Nota: in USE24-XD ogni gruppo ha esattamente un nodo (root_id ==
    node_id per ogni record), quindi la stella degenera in un nodo
    isolato. Questo e' il fenomeno di Concept Drift documentato nel
    Capitolo 4 della tesi: la disinformazione del 2024 si manifesta
    prevalentemente come broadcast di post singoli, non come catena
    di repliche strutturate come nel 2016.

    Args:
        df: Subset del DataFrame relativo a una singola cascata,
            privo della colonna parent_id.

    Returns:
        Copia del DataFrame con la colonna parent_id aggiunta.
        Il valore e' None per la radice, root_id per tutti gli altri.
    """
    df = df.copy()
    root_val = df["root_id"].iloc[0]
    df["parent_id"] = df["node_id"].apply(
        lambda nid: None if nid == root_val else root_val
    )
    return df


def create_bigcn_data(df_group: pd.DataFrame) -> Data | None:
    """Converte una singola cascata in un oggetto Data per Bi-GCN.

    Gestisce dinamicamente due topologie eterogenee in base alla
    presenza o assenza della colonna parent_id nel DataFrame:

    Topologia ad Albero (PHEME 2016):
        parent_id e' presente nel dataset. Gli archi riflettono la
        struttura reale della conversazione Twitter: un tweet radice
        genera risposte, le risposte generano ulteriori risposte, e
        cosi' via in forma di albero gerarchico a profondita' variabile.

    Topologia a Punti Isolati (USE24 2024):
        parent_id e' assente. Viene sintetizzato da _inject_star_topology:
        tutti i nodi puntano alla radice (stella). In USE24-XD ogni
        cascata ha tipicamente un unico nodo, rendendo la stella un
        nodo isolato con edge_index vuoto.

    Per Bi-GCN, gli archi vengono sempre costruiti in forma bidirezionale:
        - edge_index_td (Top-Down): parent -> child
          Modella la propagazione dell'informazione verso il basso.
        - edge_index_bu (Bottom-Up): child -> parent
          Modella la risposta dell'audience verso la radice.

    I self-loop (p_idx == c_idx) vengono esclusi perche' PyTorch
    Geometric li aggiunge automaticamente tramite add_self_loops()
    durante la convoluzione GCN, evitando duplicazioni.

    Args:
        df_group: pd.DataFrame di una singola cascata (stesso root_id)
                  con colonne: node_id, parent_id (opzionale),
                  root_id, embedding_bin, status (o label).

    Returns:
        torch_geometric.data.Data con attributi:
          x             : feature matrix [N, 768] float32
          edge_index    : alias di edge_index_td per compatibilita' PyG
          edge_index_td : archi top-down [2, E]
          edge_index_bu : archi bottom-up [2, E]
          y             : label scalare [1] long
          root_id       : identificatore stringa della cascata
          num_nodes     : numero di nodi N
        Oppure None se il DataFrame e' vuoto o malformato.
    """
    if df_group is None or df_group.empty:
        return None

    # -- Iniezione parent_id sintetico per topologia a stella -----------------
    if "parent_id" not in df_group.columns:
        df_group = _inject_star_topology(df_group)
    else:
        # Copia difensiva per evitare SettingWithCopyWarning di Pandas
        df_group = df_group.copy()

    # -- 1. Mapping node_id -> indice intero contiguo -------------------------
    # sorted() garantisce determinismo tra esecuzioni diverse, fondamentale
    # per la riproducibilita' degli esperimenti di Continual Learning (EWC).
    unique_ids = sorted(df_group["node_id"].dropna().unique().tolist())
    if not unique_ids:
        return None

    node_to_idx: dict[str, int] = {
        nid: i for i, nid in enumerate(unique_ids)
    }
    n_nodes = len(node_to_idx)

    # -- 2. Decodifica embedding Float16 -> tensore Float32 [N, 768] ----------
    x_np = np.zeros((n_nodes, EMBED_DIM), dtype=np.float32)
    df_indexed = df_group.set_index("node_id")

    for nid, idx in node_to_idx.items():
        if nid not in df_indexed.index:
            warnings.warn(
                f"Nodo {nid} assente nell'indice. "
                "Embedding inizializzato a zero."
            )
            continue

        raw = df_indexed.loc[nid, "embedding_bin"]

        # Gestione duplicati accidentali nell'indice Pandas
        if isinstance(raw, pd.Series):
            raw = raw.iloc[0]

        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue

        try:
            emb_f16 = np.frombuffer(raw, dtype=np.float16)
            if emb_f16.shape[0] == EMBED_DIM:
                x_np[idx] = emb_f16.astype(np.float32)
        except (ValueError, TypeError):
            pass

    x = torch.from_numpy(x_np)

#### NOTA: VECCHIA VERSIONE DEL BLOCCO 3, NON USARE
#    # -- 3. Costruzione edge_index TD e BU ------------------------------------
#    src_td: list[int] = []
#    dst_td: list[int] = []
#    src_bu: list[int] = []
#    dst_bu: list[int] = []
#
#    for _, row in df_group.iterrows():
#        child_id  = row["node_id"]
#        parent_id = row["parent_id"]
#
#        if pd.isna(parent_id):
#            continue  # nodo radice: nessun arco entrante
#
#        if child_id not in node_to_idx or parent_id not in node_to_idx:
#            # parent_id non presente nel mapping: ramo tagliato dall'API
#            continue
#
#        p_idx = node_to_idx[parent_id]
#        c_idx = node_to_idx[child_id]
#
#        # Esclusione self-loop: PyG li aggiunge via add_self_loops()
#        if p_idx == c_idx:
#            continue
#
#        src_td.append(p_idx)
#        dst_td.append(c_idx)
#        src_bu.append(c_idx)
#        dst_bu.append(p_idx)
#
#    if src_td:
#        edge_index_td = torch.tensor(
#            [src_td, dst_td], dtype=torch.long
#        )
#        edge_index_bu = torch.tensor(
#            [src_bu, dst_bu], dtype=torch.long
#        )
#    else:
#        # Nodo isolato o cascata a singolo tweet: tensori vuoti validi
#        edge_index_td = torch.zeros((2, 0), dtype=torch.long)
#        edge_index_bu = torch.zeros((2, 0), dtype=torch.long)

#### NOTA: NUOVA VERSIONE DEL BLOCCO 3, CON .flip(0)
# -- 3. Costruzione edge_index TD, poi trasposizione vettoriale per BU ---
    src_td: list[int] = []
    dst_td: list[int] = []

    for _, row in df_group.iterrows():
        child_id  = row["node_id"]
        parent_id = row["parent_id"]

        if pd.isna(parent_id):
            continue  # nodo radice: nessun arco entrante

        if child_id not in node_to_idx or parent_id not in node_to_idx:
            # parent_id non presente nel mapping: ramo tagliato dall'API
            continue

        p_idx = node_to_idx[parent_id]
        c_idx = node_to_idx[child_id]

        # Esclusione self-loop: PyG li aggiunge via add_self_loops()
        if p_idx == c_idx:
            continue

        src_td.append(p_idx)
        dst_td.append(c_idx)

    if src_td:
        edge_index_td = torch.tensor(
            [src_td, dst_td], dtype=torch.long
        )
        # Trasposizione vettorizzata: BU e' TD con sorgente e destinazione
        # invertite. flip(0) scambia le due righe del tensore in un'unica
        # operazione O(E), senza ricostruire liste Python o ricalcolare il grafo.
        edge_index_bu = edge_index_td.flip(0)
    else:
        # Nodo isolato o cascata a singolo tweet: tensori vuoti validi
        edge_index_td = torch.zeros((2, 0), dtype=torch.long)
        edge_index_bu = torch.zeros((2, 0), dtype=torch.long)

    # -- 4. Assegnazione label globale tramite moda ----------------------------
    # Ricerca adattiva della colonna label: supporta 'status' (PHEME/USE24)
    # e 'label' (LIAR), garantendo compatibilita' con tutti i dataset.
    if "status" in df_group.columns:
        label_col = "status"
    elif "label" in df_group.columns:
        label_col = "label"
    else:
        label_col = None

    if label_col:
        raw_labels = (
            df_group[label_col]
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
            .tolist()
        )
        mapped = [LABEL_MAP.get(lbl, -1) for lbl in raw_labels]
        valid  = [m for m in mapped if m >= 0]
        y_val  = max(set(valid), key=valid.count) if valid else -1
    else:
        y_val = -1

    y = torch.tensor([y_val], dtype=torch.long)

    # -- 5. Costruzione oggetto Data ------------------------------------------
    data               = Data(x=x, edge_index=edge_index_td, y=y)
    data.edge_index_td = edge_index_td
    data.edge_index_bu = edge_index_bu
    data.root_id       = df_group["root_id"].iloc[0]
    data.num_nodes     = n_nodes

    return data


# ##############################################################################
# ### FASE 2: PIPELINE DI GENERAZIONE GRAFI (process_and_save_graphs)       ###
# ##############################################################################

def process_and_save_graphs(
    parquet_path: str,
    output_path:  str,
    dataset_name: str,
) -> list[Data]:
    """Carica un Parquet vettorizzato e genera la lista di grafi PyG.

    Strategia di salvataggio in file unico (.pt):
        PHEME ha circa 6.500 cascate; USE24 circa 97.000 nodi. Salvare
        un file .pt per ogni grafo genererebbe decine di migliaia di
        micro-file su Google Drive, degradando le performance di I/O
        in modo critico (Drive impone rate-limit sulle operazioni sui
        metadata). La soluzione adottata e' caricare l'intero Parquet
        in RAM Pandas (il file pesa ~170 MB, abbondantemente gestibile
        nei 12 GB di RAM di Colab), generare tutti i grafi in memoria
        e serializzarli in un unico archivio .pt tramite torch.save().
        Questo file verra' caricato in un'unica operazione durante il
        training Bi-GCN, azzerando il costo di I/O per mini-batch.

    Args:
        parquet_path: Percorso del Parquet vettorizzato FP16.
        output_path:  Percorso del file .pt di output.
        dataset_name: Nome del dataset per i log di avanzamento.

    Returns:
        Lista di oggetti Data salvata su disco. Ogni elemento
        corrisponde a una cascata (root_id) del dataset originale.
        Le cascate vuote o malformate vengono silenziosamente escluse.
    """
    print(f"Caricamento Parquet di {dataset_name} in RAM...")
    df = pd.read_parquet(parquet_path)

    print(f"Raggruppamento cascate per root_id ({dataset_name})...")
    grouped = df.groupby("root_id", sort=False)

    graphs: list[Data] = []
    error_count        = 0

    print(f"Generazione grafi ({dataset_name})...")
    for _, group in tqdm(grouped, desc=dataset_name, unit="grafo"):
        try:
            graph_data = create_bigcn_data(group)
            if graph_data is not None and graph_data.num_nodes > 0:
                graphs.append(graph_data)
        except Exception:
            error_count += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graphs, output_path)

    print(
        f"Generazione completata per {dataset_name}: "
        f"{len(graphs)} grafi salvati in {output_path}."
    )
    if error_count > 0:
        print(
            f"Cascate scartate per errori in {dataset_name}: "
            f"{error_count}."
        )

    return graphs


# ##############################################################################
# ### FASE 3: VALIDAZIONE TOPOLOGICA (validate_topology)                    ###
# ##############################################################################

def validate_topology(
    filepath:     str,
    dataset_name: str,
    n_sample:     int = 1,
) -> None:
    """Valida struttura, topologia e distribuzione label di un file .pt.

    Consolida le funzioni inspect_graphs, final_sanity_check e
    check_labels in un unico report strutturato. Le verifiche eseguite
    sono suddivise in tre livelli:

    Livello 1 — Struttura:
        Caricamento del file, conteggio totale dei grafi, dimensioni
        medie e massime per nodi e archi.

    Livello 2 — Topologia:
        Verifica che il numero di archi Top-Down e Bottom-Up sia
        identico per tutti i grafi (proprieta' necessaria per Bi-GCN).
        Rileva automaticamente la topologia prevalente:
          - Albero/Mista: archi != (num_nodes - 1)
            Tipico di PHEME, dove le conversazioni hanno profondita'
            variabile e branching factor > 1.
          - Stella/Nodo isolato: archi == (num_nodes - 1)
            Tipico di USE24, dove ogni post e' radice di sé stesso.
            La stella degenera in nodo isolato quando num_nodes == 1,
            confermando il Concept Drift topologico 2016 -> 2024.

    Livello 3 — Label:
        Distribuzione delle classi su tutti i grafi. Un conteggio
        elevato di label -1 indica voci non presenti in LABEL_MAP
        e richiede un aggiornamento del dizionario.

    Args:
        filepath:     Percorso del file .pt contenente la lista di Data.
        dataset_name: Nome del dataset per i log del report.
        n_sample:     Numero di grafi campione da ispezionare in
                      dettaglio (default: 1).

    Returns:
        None. Stampa un report testuale strutturato.
    """
    SEP = "=" * 55
    print(f"\n{SEP}")
    print(f"  Validazione topologica: {dataset_name}")
    print(SEP)

    # -- Caricamento ------------------------------------------------------------
    # weights_only=False: necessario per PyTorch >= 2.6, che ha inasprito
    # le restrizioni di sicurezza su torch.load() per oggetti personalizzati
    # come torch_geometric.data.Data (non un semplice tensore).
    try:
        graphs: list[Data] = torch.load(filepath, weights_only=False)
    except Exception as exc:
        print(f"  ERRORE caricamento: {exc}")
        return

    total = len(graphs)
    if total == 0:
        print("  Nessun grafo presente nel file.")
        return

    print(f"  Totale cascate (grafi): {total}")

    # -- Livello 1: Statistiche strutturali ------------------------------------
    nodes_list  = [g.num_nodes for g in graphs]
    edges_td    = [g.edge_index_td.shape[1] for g in graphs]
    edges_bu    = [g.edge_index_bu.shape[1] for g in graphs]

    print(
        f"  Nodi per cascata  — media: "
        f"{sum(nodes_list)/total:.2f}  "
        f"max: {max(nodes_list)}"
    )
    print(
        f"  Archi Top-Down    — media: "
        f"{sum(edges_td)/total:.2f}  "
        f"max: {max(edges_td)}"
    )

    # -- Livello 2: Integrità bidirezionale e topologia ------------------------
    if sum(edges_td) == sum(edges_bu):
        print(
            "  Integrita' bidirezionale: "
            "archi TD e BU corrispondenti [OK]."
        )
    else:
        delta = abs(sum(edges_td) - sum(edges_bu))
        print(
            f"  ANOMALIA bidirezionale: "
            f"delta TD/BU = {delta} archi."
        )

    # Rilevamento topologia sul campione 0
    sample_0 = graphs[0]
    if sample_0.num_nodes > 1:
        expected_star = sample_0.num_nodes - 1
        actual_edges  = sample_0.edge_index_td.shape[1]
        if actual_edges == expected_star:
            topo_label = "STELLA / NODO ISOLATO (Concept Drift 2024)"
        else:
            topo_label = "ALBERO / MISTA (Struttura gerarchica 2016)"
    else:
        topo_label = "NODO SINGOLO (cascata a un elemento)"

    print(f"  Topologia prevalente: {topo_label}")

    # -- Ispezione campioni ----------------------------------------------------
    print(f"\n  Ispezione dettagliata ({n_sample} campione/i):")
    for i in range(min(n_sample, total)):
        g = graphs[i]
        print(
            f"    Grafo {i}: "
            f"nodi={g.num_nodes}  "
            f"archi_td={g.edge_index_td.shape[1]}  "
            f"x={list(g.x.shape)}  "
            f"y={g.y.item()}  "
            f"dtype={g.x.dtype}"
        )

    # -- Livello 3: Distribuzione label ----------------------------------------
    label_dist: dict[int, int] = {}
    for g in graphs:
        lbl = g.y.item()
        label_dist[lbl] = label_dist.get(lbl, 0) + 1

    print("\n  Distribuzione label:")
    label_names = {0: "Fake/Rumor", 1: "True/Non-Rumor",
                   2: "Unverified", -1: "Non mappata"}
    for lbl in sorted(label_dist.keys()):
        count  = label_dist[lbl]
        pct    = count / total * 100
        name   = label_names.get(lbl, f"Classe {lbl}")
        print(f"    {name} ({lbl}): {count} ({pct:.1f}%)")

    if label_dist.get(-1, 0) > 0:
        print(
            "  ATTENZIONE: label -1 rilevate. "
            "Verificare LABEL_MAP per le voci mancanti."
        )

    print(f"{SEP}\n")


# ##############################################################################
# ### ENTRY POINT: ESECUZIONE SEQUENZIALE DELLA PIPELINE                    ###
# ##############################################################################

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- Configurazione job: (parquet_input, output_pt, nome_dataset) ---------
    pipeline_jobs = [
        (
            IN_PHEME,
            f"{OUTPUT_DIR}/PHEME_graphs_list.pt",
            "PHEME",
        ),
        (
            IN_USE24,
            f"{OUTPUT_DIR}/USE24_graphs_list.pt",
            "USE24",
        ),
    ]

    # -- Generazione grafi per tutti i dataset ---------------------------------
    for parquet_path, output_path, name in pipeline_jobs:
        process_and_save_graphs(
            parquet_path = parquet_path,
            output_path  = output_path,
            dataset_name = name,
        )

    # -- Validazione topologica su tutti i dataset ----------------------------
    print("\nAvvio validazione topologica...")
    for _, output_path, name in pipeline_jobs:
        validate_topology(
            filepath     = output_path,
            dataset_name = name,
            n_sample     = 1,
        )

    print("Pipeline di feature extraction topologica completata.")
