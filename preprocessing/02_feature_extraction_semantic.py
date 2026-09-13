# =============================================================================
# FILE:    feature_extraction_semantic.py
# SCOPO:   Estrazione delle feature semantiche tramite BERTweet (vinai/
#          bertweet-base). Produce un embedding [CLS] Float16 per ogni
#          nodo dei dataset LIAR, PHEME e USE24, salvato in formato
#          Parquet binario (embedding_bin) per la successiva fase Bi-GCN.
# DIPENDENZE: pyspark==3.5.0, transformers, torch, emoji==0.6.0, numpy
# MODULO:  2 di 3 — da eseguire dopo etl_pipeline.py
# =============================================================================


# ##############################################################################
# ### FASE 0: IMPORT E COSTANTI GLOBALI                                      ###
# ##############################################################################

import re
import os
import unicodedata

import emoji
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import BinaryType, StringType

# -- Modello BERTweet ----------------------------------------------------------
BERTWEET_MODEL_ID = "vinai/bertweet-base"
EMBED_DIM         = 768      # Dimensione del vettore [CLS] di BERTweet
MAX_TOKEN_LEN     = 128      # Lunghezza massima sequenza (tweet <= 280 char)

# -- Percorsi Parquet di input (output di etl_pipeline.py) --------------------
DRIVE_BASE    = "/content/drive/MyDrive/Tesi/Parquet_Finali"
IN_LIAR       = f"{DRIVE_BASE}/LIAR_Full_Benchmark.parquet"
IN_PHEME      = f"{DRIVE_BASE}/PHEME_Unified.parquet"
IN_USE24      = f"{DRIVE_BASE}/USE24_Unified.parquet"

# -- Percorsi Parquet di output (Float16 quantizzati) -------------------------
OUT_LIAR      = f"{DRIVE_BASE}/LIAR_Vectorized_FP16.parquet"
OUT_PHEME     = f"{DRIVE_BASE}/PHEME_Vectorized_FP16.parquet"
OUT_USE24     = f"{DRIVE_BASE}/USE24_Vectorized_FP16.parquet"

# -- Conteggi attesi (per la validazione post-elaborazione) -------------------
EXPECTED_COUNTS = {
    "LIAR":  12836,
    "PHEME": 103212,
    "USE24": 97696,
}

# -- Numero di partizioni Parquet per dataset ---------------------------------
# Valore calibrato per Colab: evita file troppo piccoli (overhead I/O)
# o troppo grandi (timeout Drive). Circa 5 MB/partizione su GPU T4.
PARTITIONS = {
    "LIAR":  5,
    "PHEME": 20,
    "USE24": 20,
}


# ##############################################################################
# ### FASE 1: CONFIGURAZIONE SPARK E APACHE ARROW                            ###
# ##############################################################################

def create_spark_session_arrow(app_name: str = "BERTweet_Vectorization") \
        -> SparkSession:
    """Crea una SparkSession con supporto Apache Arrow abilitato.

    Apache Arrow e' il formato di memoria colonnare condiviso tra Spark
    e il runtime Python. Senza Arrow, ogni chiamata a una Pandas UDF
    richiede la serializzazione/deserializzazione riga per riga tramite
    Pickle, con un overhead di 10-50x rispetto al batch processing.
    Con Arrow abilitato, Spark trasferisce interi batch di dati al
    processo Python in un unico blocco di memoria contigua, eliminando
    la maggior parte del costo di interprocess communication.

    Il parametro maxRecordsPerBatch=64 e' calibrato per mantenere i
    batch di embedding BERTweet entro la VRAM disponibile su GPU T4
    (16 GB): 64 testi * 128 token * 768 dim * 2 byte (Float16) ~= 12 MB.

    Args:
        app_name: Nome dell'applicazione visualizzato nella Spark UI.

    Returns:
        SparkSession attiva con Arrow e batch size configurati.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config(
            "spark.sql.execution.arrow.pyspark.enabled", "true"
        )
        .config(
            "spark.sql.execution.arrow.maxRecordsPerBatch", "64"
        )
        .getOrCreate()
    )
    return spark


# ##############################################################################
# ### FASE 2: PRE-PROCESSING TESTUALE (PANDAS UDF)                           ###
# ##############################################################################

def clean_text_for_bertweet(text: str) -> str:
    """Normalizza un testo grezzo secondo le convenzioni di BERTweet.

    BERTweet e' stato pre-addestrato su tweet gia' normalizzati: usare
    token diversi da quelli del vocabolario nativo degraderebbe la
    qualita' dell'embedding per effetto OOV (Out-Of-Vocabulary). Le
    convenzioni adottate sono allineate al preprocessore originale
    TweetTokenizer di BERTweet (Nguyen et al., 2020):

    - Menzioni utente  -> @USER   (anonimizzazione e token nativo)
    - URL              -> HTTPURL (presenza link come segnale semantico)
    - Emoji            -> :token: (segnale emozionale preservato)
    - Caratteri Cc/Cf  -> rimossi (controllo, BOM, zero-width)

    Args:
        text: Stringa di testo grezza proveniente dal dataset.

    Returns:
        Stringa normalizzata pronta per la tokenizzazione BERTweet,
        oppure stringa vuota se l'input e' nullo o non testuale.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Normalizzazione Unicode NFC: risolve varianti grafiche identiche
    # codificate diversamente (es. e + combining accent vs e-accentata).
    text = unicodedata.normalize("NFC", text)

    # URL -> HTTPURL: il segnale della presenza di un link e' informativo
    # per la fake news detection (i rumor tendono a linkare fonti dubbie).
    text = re.sub(
        r"(?:https?://|www\.)\S+", "HTTPURL", text, flags=re.IGNORECASE
    )

    # Menzioni -> @USER: anonimizza il mittente e usa il token nativo
    # del vocabolario BERTweet.
    text = re.sub(r"@\w+", "@USER", text)

    # Emoji -> descrizione testuale (:fire:, :thumbs_up:, ecc.)
    # La libreria emoji.demojize preserva il contenuto semantico
    # dell'emoji che sarebbe altrimenti irrecuperabile per il modello.
    text = emoji.demojize(text, delimiters=(" :", ": "))

    # Rimozione caratteri non stampabili (categorie Unicode Cc, Cf, Cs):
    # null bytes, BOM, caratteri di formattazione invisibili.
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"Cc", "Cf", "Cs"}
    )

    return re.sub(r"\s+", " ", text).strip()


@pandas_udf(StringType())
def clean_text_udf(series: pd.Series) -> pd.Series:
    """Pandas UDF scalare per la pulizia testuale su batch Arrow.

    Wrappa clean_text_for_bertweet per l'esecuzione distribuita
    su Spark con serializzazione Arrow. Opera su batch di 64 record
    (configurato in create_spark_session_arrow) anziche' riga per riga,
    riducendo l'overhead di interprocess communication del 90% rispetto
    alle Row UDF standard.

    Args:
        series: pd.Series di stringhe grezze (campo content_text).

    Returns:
        pd.Series di stringhe normalizzate secondo standard BERTweet.
    """
    return series.apply(clean_text_for_bertweet)


# ##############################################################################
# ### FASE 3: FEATURE EXTRACTION SEMANTICA (BERTWEET + PANDAS UDF)           ###
# ##############################################################################

def _load_bertweet_model(
    model_id: str,
    device: str
) -> tuple[AutoTokenizer, AutoModel]:
    """Carica il tokenizer e il modello BERTweet sul dispositivo specificato.

    Utilizza normalization=True nel tokenizer per garantire la coerenza
    con il preprocessing nativo del modello. Il modello viene caricato
    in modalita' eval() per disabilitare Dropout e BatchNorm, necessario
    per la riproducibilita' degli embedding in inferenza.

    Args:
        model_id: Identificatore HuggingFace del modello.
        device:   "cuda" o "cpu".

    Returns:
        Tupla (tokenizer, model) pronti per l'inferenza.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, normalization=True
    )
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    return tokenizer, model


def build_bertweet_udf(
    tokenizer: AutoTokenizer,
    model:     AutoModel,
    device:    str,
    max_len:   int = MAX_TOKEN_LEN,
    embed_dim: int = EMBED_DIM
):
    """Costruisce la Pandas UDF per l'estrazione dell'embedding [CLS].

    La UDF restituisce l'embedding del token [CLS] di BERTweet, ovvero
    il vettore di rappresentazione aggregata della sequenza di input
    (dimensione: 768). Questo vettore viene quantizzato in Float16
    (16 bit per valore invece di 32) e serializzato in byte tramite
    .tobytes(). La quantizzazione Float16 riduce del 50% l'occupazione
    su disco e in RAM, con una perdita di precisione trascurabile per
    l'inferenza (l'errore massimo di rappresentazione Float16 e'
    dell'ordine di 1e-3, inferiore alla varianza tipica degli embedding).

    La UDF sfrutta Arrow per ricevere i batch dal driver Spark senza
    overhead di serializzazione, e torch.no_grad() per disabilitare
    il calcolo del grafo computazionale di PyTorch durante l'inferenza,
    riducendo il consumo di VRAM del 30-40%.

    Args:
        tokenizer: Tokenizer BERTweet gia' caricato.
        model:     Modello BERTweet gia' caricato e in modalita' eval.
        device:    Dispositivo di calcolo ("cuda" o "cpu").
        max_len:   Lunghezza massima della sequenza tokenizzata.
        embed_dim: Dimensione attesa del vettore [CLS].

    Returns:
        Pandas UDF (BinaryType) registrabile su un DataFrame Spark.
    """
    @pandas_udf(BinaryType())
    def get_bertweet_binary_udf(texts: pd.Series) -> pd.Series:
        inputs = tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Estrazione token [CLS] (indice 0): rappresentazione
            # aggregata dell'intera sequenza secondo BERTweet.
            embeddings = (
                outputs.last_hidden_state[:, 0, :]
                .cpu()
                .numpy()
            )

        # Quantizzazione Float32 -> Float16 e serializzazione in byte.
        # .tobytes() produce un array di (768 * 2) = 1536 byte per nodo.
        return pd.Series([
            emb.astype(np.float16).tobytes() for emb in embeddings
        ])

    return get_bertweet_binary_udf


# ##############################################################################
# ### FASE 4: PIPELINE DI VECTORIZATION SUI DATASET                         ###
# ##############################################################################

def vectorize_dataset(
    spark:        SparkSession,
    input_path:   str,
    output_path:  str,
    dataset_name: str,
    clean_udf,
    embed_udf,
    n_partitions: int,
) -> None:
    """Applica pulizia testuale ed estrazione embedding a un dataset Parquet.

    Esegue la pipeline completa in tre passi:
    1. Caricamento del Parquet normalizzato (output di etl_pipeline.py).
    2. Applicazione della clean_text_udf Arrow-ottimizzata.
    3. Applicazione della embed_udf BERTweet su GPU con quantizzazione FP16.
    4. Salvataggio del risultato con repartitioning controllato.

    Il repartitioning prima del salvataggio evita di generare centinaia
    di micro-file Parquet (uno per task Spark), che degraderebbero le
    performance di lettura nella fase successiva di training Bi-GCN.

    Args:
        spark:        SparkSession attiva con Arrow abilitato.
        input_path:   Percorso del Parquet di input (non vettorizzato).
        output_path:  Percorso del Parquet di output (vettorizzato FP16).
        dataset_name: Nome del dataset per i log di avanzamento.
        clean_udf:    Pandas UDF per la normalizzazione testuale.
        embed_udf:    Pandas UDF per l'estrazione embedding BERTweet.
        n_partitions: Numero di partizioni Parquet di output.

    Returns:
        None
    """
    df = spark.read.parquet(input_path)
    print(
        f"Avvio vectorization su {dataset_name} "
        f"({df.count()} record)."
    )

    df_vectorized = (
        df
        .withColumn(
            "content_text_clean", clean_udf(col("content_text"))
        )
        .withColumn(
            "embedding_bin", embed_udf(col("content_text_clean"))
        )
    )

    (
        df_vectorized
        .repartition(n_partitions)
        .write
        .mode("overwrite")
        .parquet(output_path)
    )
    print(f"Vectorization completata per {dataset_name}: {output_path}")


# ##############################################################################
# ### FASE 5: VALIDAZIONE INTEGRITA' E QUALITA' SEMANTICA                   ###
# ##############################################################################

def validate_parquet_integrity(
    spark:           SparkSession,
    datasets_info:   dict,
) -> None:
    """Verifica l'integrita' strutturale dei Parquet vettorizzati.

    Per ogni dataset controlla: esistenza su disco, dimensione in MB,
    leggibilita' da Spark e corrispondenza del conteggio record con
    il valore atteso. Un'anomalia nel conteggio indica una perdita di
    dati durante la vectorization (tipicamente causata da OOM su GPU
    o timeout del driver Spark).

    Args:
        spark:         SparkSession attiva.
        datasets_info: Dizionario con struttura:
                       { "NOME": {"path": str, "expected_count": int} }

    Returns:
        None. Stampa un report per ogni dataset.
    """
    for name, info in datasets_info.items():
        path     = info["path"]
        expected = info["expected_count"]

        print(f"\n{'=' * 55}")
        print(f"  Verifica integrita': {name}")
        print(f"{'=' * 55}")

        # -- Dimensione su disco ---------------------------------------------
        if not os.path.exists(path):
            print(f"  ERRORE: percorso non trovato: {path}")
            continue

        size_bytes = sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, filenames in os.walk(path)
            for fn in filenames
        )
        print(f"  Dimensione su disco: {size_bytes / (1024**2):.2f} MB")

        # -- Lettura e conteggio Spark ----------------------------------------
        try:
            df           = spark.read.parquet(path)
            actual_count = df.count()

            if actual_count == expected:
                print(
                    f"  Conteggio record: {actual_count} / {expected}"
                    " [CORRETTO]"
                )
            else:
                delta = expected - actual_count
                print(
                    f"  Conteggio record: {actual_count} / {expected}"
                    f" [ANOMALIA: delta = {delta}]"
                )

        except Exception as exc:
            print(f"  ERRORE lettura Parquet: {exc}")


def validate_embedding_quality(
    spark:         SparkSession,
    datasets_info: dict,
    n_samples:     int = 5,
) -> None:
    """Verifica la qualita' semantica degli embedding BERTweet campionati.

    Per ogni dataset estrae n_samples embedding casuali e verifica:
    - Dimensione: il vettore deve avere esattamente 768 componenti.
    - Assenza di NaN: valori Not-a-Number corrompono il calcolo GNN.
    - Assenza di Inf: valori infiniti causano overflow nei layer lineari.
    - Non-zero: un vettore tutto zero indica un errore di serializzazione
      o un input vuoto non filtrato.

    Stampa inoltre le statistiche descrittive (media, deviazione std)
    per ciascun campione: una deviazione std dell'ordine di 0.3-0.5
    indica che il modello ha estratto rappresentazioni semanticamente
    differenziate (vettori "vivi"). Valori prossimi a zero indicano
    collasso delle rappresentazioni.

    Args:
        spark:         SparkSession attiva.
        datasets_info: Dizionario con struttura:
                       { "NOME": {"path": str, "expected_count": int} }
        n_samples:     Numero di campioni casuali per dataset.

    Returns:
        None. Stampa un report statistico per ogni dataset.
    """
    for name, info in datasets_info.items():
        path = info["path"]

        print(f"\n{'=' * 55}")
        print(f"  Analisi qualita' semantica: {name}")
        print(f"{'=' * 55}")

        try:
            df = spark.read.parquet(path)
            samples = (
                df
                .select("content_text_clean", "embedding_bin")
                .sample(False, 0.1)
                .limit(n_samples)
                .collect()
            )
        except Exception as exc:
            print(f"  ERRORE lettura campioni: {exc}")
            continue

        all_valid = True

        for i, row in enumerate(samples):
            vec = np.frombuffer(row.embedding_bin, dtype=np.float16)

            is_768   = len(vec) == EMBED_DIM
            has_nan  = bool(np.isnan(vec).any())
            has_inf  = bool(np.isinf(vec).any())
            is_zero  = not bool(np.any(vec))

            integrity_ok = is_768 and not has_nan \
                           and not has_inf and not is_zero

            if not integrity_ok:
                all_valid = False
                print(
                    f"  Campione {i + 1} [ANOMALIA] "
                    f"dim={len(vec)} NaN={has_nan} "
                    f"Inf={has_inf} Zero={is_zero}"
                )
                print(
                    f"  Testo: "
                    f"{row.content_text_clean[:60]}..."
                )
            else:
                media   = float(vec.mean())
                dev_std = float(vec.std())
                print(
                    f"  Campione {i + 1} [OK] "
                    f"media={media:.4f} "
                    f"dev_std={dev_std:.4f} | "
                    f"{row.content_text_clean[:50]}..."
                )

        status_label = "TUTTI I CAMPIONI INTEGRI" \
            if all_valid else "ANOMALIE RILEVATE"
        print(f"  Risultato: {status_label}")


# ##############################################################################
# ### ENTRY POINT: ESECUZIONE SEQUENZIALE DELLA PIPELINE                    ###
# ##############################################################################

if __name__ == "__main__":

    # 1. Configurazione runtime
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo di calcolo rilevato: {device}.")

    spark = create_spark_session_arrow()
    print("Configurazione Spark con Apache Arrow completata.")

    # 2. Caricamento modello BERTweet
    print(f"Caricamento modello {BERTWEET_MODEL_ID}...")
    tokenizer, model = _load_bertweet_model(BERTWEET_MODEL_ID, device)
    print("Modello BERTweet caricato in modalita' eval.")

    # 3. Costruzione UDF (le UDF catturano tokenizer e model per closure)
    embed_udf = build_bertweet_udf(tokenizer, model, device)

    # 4. Vectorization sequenziale dei tre dataset
    pipeline_jobs = [
        (IN_LIAR,  OUT_LIAR,  "LIAR",  PARTITIONS["LIAR"]),
        (IN_PHEME, OUT_PHEME, "PHEME", PARTITIONS["PHEME"]),
        (IN_USE24, OUT_USE24, "USE24", PARTITIONS["USE24"]),
    ]

    for in_path, out_path, name, n_parts in pipeline_jobs:
        vectorize_dataset(
            spark        = spark,
            input_path   = in_path,
            output_path  = out_path,
            dataset_name = name,
            clean_udf    = clean_text_udf,
            embed_udf    = embed_udf,
            n_partitions = n_parts,
        )

    # 5. Validazione finale
    validation_cfg = {
        "LIAR":  {"path": OUT_LIAR,  "expected_count": EXPECTED_COUNTS["LIAR"]},
        "PHEME": {"path": OUT_PHEME, "expected_count": EXPECTED_COUNTS["PHEME"]},
        "USE24": {"path": OUT_USE24, "expected_count": EXPECTED_COUNTS["USE24"]},
    }

    print("\nAvvio validazione integrita' Parquet...")
    validate_parquet_integrity(spark, validation_cfg)

    print("\nAvvio analisi qualita' semantica embedding...")
    validate_embedding_quality(spark, validation_cfg, n_samples=5)

    spark.stop()
    print("\nPipeline di feature extraction semantica completata.")
