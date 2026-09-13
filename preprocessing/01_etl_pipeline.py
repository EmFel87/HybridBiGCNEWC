# =============================================================================
# FILE:    etl_pipeline.py
# SCOPO:   Pipeline ETL per il caricamento, la normalizzazione e il
#          salvataggio in formato Parquet dei dataset PHEME, LIAR e USE24.
#          Modulo 1 di 3 del progetto "Fake News Detection: 2016 vs 2024".
# DIPENDENZE: pyspark==3.5.0, openjdk-11
# =============================================================================


# ##############################################################################
# ### FASE 0: IMPORT E COSTANTI GLOBALI                                      ###
# ##############################################################################

import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType
)

# -- Percorsi Google Drive (modificare in base alla propria struttura) ---------
DRIVE_BASE     = "/content/drive/MyDrive/Tesi"
PHEME_GLOB     = f"{DRIVE_BASE}/Dataset_2016/pheme-rnr-dataset/*/*/*/*.json"
LIAR_BASE      = f"{DRIVE_BASE}/LIAR/"
USE24_PATH     = f"{DRIVE_BASE}/Dataset_2024/USE24_XD_dataset.csv"

PARQUET_BASE   = f"{DRIVE_BASE}/Parquet_Finali"
OUT_PHEME      = f"{PARQUET_BASE}/PHEME_Unified.parquet"
OUT_LIAR       = f"{PARQUET_BASE}/LIAR_Full_Benchmark.parquet"
OUT_USE24      = f"{PARQUET_BASE}/USE24_Unified.parquet"

# -- Colonne di misinformazione nel dataset USE24-XD --------------------------
USE24_LABEL_COLS = [
    "Conspiracy", "Hate_Speech",
    "Sensationalism", "Speculation", "Satire"
]

# -- Schema esplicito per i file TSV del dataset LIAR -------------------------
LIAR_SCHEMA = StructType([
    StructField("id",                  StringType(), True),
    StructField("label",               StringType(), True),
    StructField("statement",           StringType(), True),
    StructField("subject",             StringType(), True),
    StructField("speaker",             StringType(), True),
    StructField("job_title",           StringType(), True),
    StructField("state_info",          StringType(), True),
    StructField("party_info",          StringType(), True),
    StructField("barely_true_counts",  StringType(), True),
    StructField("false_counts",        StringType(), True),
    StructField("half_true_counts",    StringType(), True),
    StructField("mostly_true_counts",  StringType(), True),
    StructField("pants_on_fire_counts",StringType(), True),
    StructField("context",             StringType(), True),
])


# ##############################################################################
# ### FASE 1: CONFIGURAZIONE SPARK                                           ###
# ##############################################################################

def create_spark_session(app_name: str = "FakeNewsETL") -> SparkSession:
    """Crea e restituisce una SparkSession ottimizzata per Google Colab.

    Configura il driver con limiti di memoria conservativi per evitare
    il crash del PySpark Gateway tipico dell'ambiente Colab. Usa un
    singolo executor locale poiché la GPU non e' disponibile per Spark.

    Args:
        app_name: Nome dell'applicazione Spark visualizzato nella UI.

    Returns:
        SparkSession attiva e configurata.

    Example:
        >>> spark = create_spark_session("MioProgetto")
        >>> spark.version
        '3.5.0'
    """
    spark = (
        SparkSession.builder
        .master("local[1]")
        .appName(app_name)
        .config("spark.driver.memory",      "2g")
        .config("spark.executor.memory",    "2g")
        .config("spark.driver.bindAddress", "127.0.0.1")
        # Necessario per il formato timestamp non-ISO di Twitter (PHEME)
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )
    return spark


# ##############################################################################
# ### FASE 2: CARICAMENTO E UNIFICAZIONE PHEME (2016)                       ###
# ##############################################################################

def load_pheme(spark: SparkSession, glob_path: str) -> DataFrame:
    """Carica il dataset PHEME da file JSON e lo normalizza nello Schema Unificato.

    Legge ricorsivamente tutti i file JSON del dataset PHEME tramite un
    pattern glob. La label (rumour/non-rumour) e il root_id vengono
    estratti dal percorso del file tramite regex, in modo da preservare
    la struttura del grafo di propagazione indipendentemente dal contenuto
    del singolo JSON.

    Il timestamp viene convertito dal formato Twitter standard
    ("EEE MMM dd HH:mm:ss Z yyyy") al tipo TimestampType di Spark,
    richiedendo la policy LEGACY abilitata in create_spark_session().

    Args:
        spark:     SparkSession attiva.
        glob_path: Pattern glob che punta ai JSON di PHEME.
                   Esempio: ".../pheme-rnr-dataset/*/*/*/*.json"

    Returns:
        DataFrame con schema: node_id, status, content_text,
        timestamp, parent_id, root_id, platform.
        Righe con node_id o content_text nulli vengono rimosse.

    Raises:
        AnalysisException: Se il glob_path non corrisponde ad alcun file.
    """
    df_raw = (
        spark.read.json(glob_path)
        .withColumn("file_path", F.input_file_name())
    )

    # -- Rinomina colonne con punti (incompatibili con Parquet) ---------------
    for col_name in df_raw.columns:
        if "." in col_name:
            df_raw = df_raw.withColumnRenamed(
                col_name, col_name.replace(".", "_")
            )

    df_unified = df_raw.select(
        F.col("id_str").alias("node_id"),

        # Label estratta dal percorso del file: piu' affidabile del campo JSON
        F.when(
            F.col("file_path").contains("/rumours/"), "rumour"
        ).otherwise("non-rumour").alias("status"),

        F.col("text").alias("content_text"),

        # Formato timestamp originale Twitter: "Wed Jan 07 11:07:51 +0000 2015"
        F.to_timestamp(
            F.col("created_at"), "EEE MMM dd HH:mm:ss Z yyyy"
        ).alias("timestamp"),

        F.col("in_reply_to_status_id_str").alias("parent_id"),

        # root_id = ID della conversazione, estratto dal path del file
        # Pattern: .../rumours/<ROOT_ID>/source-tweet/<ID>.json
        F.regexp_extract(
            F.col("file_path"), r"/(?:non-)?rumours/(\d+)/", 1
        ).alias("root_id"),

        F.lit("PHEME_Legacy").alias("platform"),
    )

    return df_unified.dropna(subset=["node_id", "content_text"])


# ##############################################################################
# ### FASE 3: CARICAMENTO E UNIFICAZIONE LIAR (BENCHMARK)                   ###
# ##############################################################################

def load_liar(spark: SparkSession, base_path: str) -> DataFrame:
    """Carica i tre split del dataset LIAR (train/test/valid) e li unisce.

    Ogni file TSV viene letto con lo schema LIAR_SCHEMA definito nelle
    costanti globali. I tre split vengono uniti in un unico DataFrame
    con una colonna aggiuntiva 'split' per consentire la ricostruzione
    delle partizioni originali durante la fase di training.

    LIAR non ha struttura a grafo: ogni record e' un nodo radice isolato,
    pertanto root_id e parent_id non sono inclusi nello schema unificato
    per questo dataset.

    Args:
        spark:     SparkSession attiva.
        base_path: Cartella contenente train.tsv, test.tsv, valid.tsv.

    Returns:
        DataFrame con schema: node_id, status, content_text,
        platform, split. Righe con content_text nullo vengono rimosse.

    Raises:
        AnalysisException: Se base_path non esiste o i file TSV
                           sono assenti.
    """
    splits = {
        "train": "train.tsv",
        "test":  "test.tsv",
        "valid": "valid.tsv",
    }
    frames = []

    for split_name, file_name in splits.items():
        file_path = os.path.join(base_path, file_name)

        df_split = spark.read.option("sep", "\t").csv(
            file_path, schema=LIAR_SCHEMA
        )

        df_unified = df_split.select(
            F.col("id").alias("node_id"),

            # 'label' in LIAR: false, half-true, mostly-true, ecc.
            F.col("label").alias("status"),

            F.col("statement").alias("content_text"),
            F.lit("LIAR_Benchmark").alias("platform"),

            # Colonna di split: indispensabile per la fase di training/eval
            F.lit(split_name).alias("split"),
        )
        frames.append(df_unified)

    df_full = frames[0].union(frames[1]).union(frames[2])
    return df_full.dropna(subset=["content_text"])


# ##############################################################################
# ### FASE 4: CARICAMENTO E UNIFICAZIONE USE24-XD (2024)                    ###
# ##############################################################################

def _build_status_logic(label_cols: list):
    """Costruisce la logica di priorita' per la colonna 'status' in USE24.

    USE24-XD e' un dataset multi-label: ogni post puo' avere piu' etichette
    attive simultaneamente (es. Conspiracy=1 e Sensationalism=1). Per
    ricondurlo a una classificazione single-label compatibile con il
    modello, si applica una logica di priorita' nell'ordine di label_cols.
    Se nessuna label e' attiva, il post viene classificato come "Neutral".

    Args:
        label_cols: Lista ordinata di nomi di colonna booleane (0/1).

    Returns:
        Colonna Spark (Column) con la logica when/otherwise concatenata.
    """
    # Prima condizione: la prima label della lista
    status_expr = F.when(F.col(label_cols[0]) == 1, label_cols[0])

    # Condizioni successive: concatenate in catena
    for label in label_cols[1:]:
        status_expr = status_expr.when(F.col(label) == 1, label)

    return status_expr.otherwise("Neutral")


def load_use24(spark: SparkSession, file_path: str) -> DataFrame:
    """Carica il dataset USE24-XD da CSV e lo normalizza nello Schema Unificato.

    USE24-XD contiene post di Twitter/X del 2024 con struttura a stella:
    ogni post e' un nodo radice (root_id == node_id) senza risposte
    annidate. Le metriche di engagement (retweets, likes) vengono
    preservate come feature aggiuntive per il modello Bi-GCN.

    La lettura utilizza le opzioni multiLine e escape per gestire
    correttamente i testi che contengono virgole e ritorni a capo
    all'interno del campo 'text'.

    Args:
        spark:     SparkSession attiva.
        file_path: Percorso assoluto al file USE24_XD_dataset.csv.

    Returns:
        DataFrame con schema: node_id, status, content_text, timestamp,
        platform, root_id, retweets, likes.
        Vengono rimossi i record con node_id non numerico, content_text
        o status nulli.

    Raises:
        AnalysisException: Se file_path non esiste.
    """
    df_raw = (
        spark.read
        .option("header",    "true")
        .option("multiLine", "true")
        .option("escape",    "\"")
        .option("quote",     "\"")
        .option("inferSchema", "true")
        .csv(file_path)
    )

    # -- Rinomina colonne con punti (es. "public_metrics.retweet_count") ------
    for col_name in df_raw.columns:
        if "." in col_name:
            df_raw = df_raw.withColumnRenamed(
                col_name, col_name.replace(".", "_")
            )

    status_logic = _build_status_logic(USE24_LABEL_COLS)

    df_unified = df_raw.select(
        F.col("id").cast("string").alias("node_id"),

        status_logic.alias("status"),

        F.col("text").alias("content_text"),

        # USE24-XD usa formato ISO 8601: Spark lo inferisce automaticamente
        F.to_timestamp(F.col("created_at")).alias("timestamp"),

        F.lit("X_2024_USE24").alias("platform"),

        # Struttura a stella: ogni post e' la propria radice
        F.col("id").cast("string").alias("root_id"),

        # Feature di engagement per il Bi-GCN
        F.col("public_metrics_retweet_count")
         .cast(IntegerType()).alias("retweets"),
        F.col("public_metrics_like_count")
         .cast(IntegerType()).alias("likes"),
    )

    # Rimuove artefatti testuali che hanno superato il parsing CSV
    df_clean = df_unified.filter(
        F.col("node_id").rlike("^[0-9]+$")
    )
    return df_clean.dropna(subset=["content_text", "status"])


# ##############################################################################
# ### FASE 5: SALVATAGGIO IN PARQUET                                         ###
# ##############################################################################

def save_parquet(df: DataFrame, output_path: str) -> None:
    """Salva un DataFrame in formato Parquet su Google Drive.

    Utilizza la modalita' 'overwrite' per idempotenza: esecuzioni
    successive della pipeline sovrascrivono il file esistente senza
    errori. Il formato Parquet garantisce la preservazione dei tipi
    di dato (es. TimestampType) e una compressione efficiente per
    dataset di testo.

    Args:
        df:          DataFrame Spark da persistere.
        output_path: Percorso della cartella Parquet di destinazione.

    Returns:
        None
    """
    df.write.mode("overwrite").parquet(output_path)
    print(f"Salvataggio completato: {output_path}")
    print(f"Record totali: {df.count()}")


# ##############################################################################
# ### FASE 6: VERIFICA INTEGRITA' DEI PARQUET                               ###
# ##############################################################################

def verify_parquets(spark: SparkSession, paths: dict) -> None:
    """Verifica l'integrita' dei file Parquet salvati.

    Per ogni dataset esegue: conteggio totale, rilevamento valori nulli
    sulle colonne critiche (node_id, content_text, status) e
    distribuzione delle classi. Per PHEME, aggiunge la verifica
    dell'integrita' della struttura del grafo (conteggio root_id
    univoci e root_id vuoti).

    Args:
        spark: SparkSession attiva.
        paths: Dizionario {nome_dataset: percorso_parquet}.

    Returns:
        None
    """
    CRITICAL_COLS = ["node_id", "content_text", "status"]

    for name, path in paths.items():
        print(f"\n{'=' * 55}")
        print(f"  VERIFICA: {name}")
        print(f"{'=' * 55}")

        try:
            df = spark.read.parquet(path)

            # -- Conteggio totale --------------------------------------------
            total = df.count()
            print(f"Record totali: {total}")

            # -- Valori nulli nelle colonne critiche -------------------------
            available = [c for c in CRITICAL_COLS if c in df.columns]
            null_check = df.select([
                F.count(F.when(F.col(c).isNull(), c)).alias(c)
                for c in available
            ]).collect()[0].asDict()
            print(f"Valori nulli (colonne critiche): {null_check}")

            # -- Distribuzione delle classi ----------------------------------
            print("Distribuzione classi (status):")
            df.groupBy("status").count() \
              .orderBy("count", ascending=False).show()

            # -- Verifica integrita' grafo (solo PHEME) ----------------------
            if "PHEME" in name:
                unique_roots = df.select("root_id").distinct().count()
                empty_roots = df.filter(
                    F.col("root_id").isNull() | (F.col("root_id") == "")
                ).count()
                print(f"Conversazioni (root_id) uniche: {unique_roots}")
                print(f"root_id mancanti o vuoti:       {empty_roots}")

        except Exception as exc:
            print(f"Errore nel caricamento di '{name}': {exc}")

    print(f"\n{'=' * 55}")
    print("  FINE VERIFICA")
    print(f"{'=' * 55}\n")


# ##############################################################################
# ### ENTRY POINT: ESECUZIONE SEQUENZIALE DELLA PIPELINE                    ###
# ##############################################################################

if __name__ == "__main__":

    # 1. Inizializzazione Spark
    spark = create_spark_session("FakeNews_ETL_Pipeline")
    print("Configurazione Spark completata.")

    # 2. Caricamento e salvataggio PHEME (2016)
    print("\nCaricamento PHEME 2016...")
    df_pheme = load_pheme(spark, PHEME_GLOB)
    save_parquet(df_pheme, OUT_PHEME)

    # 3. Caricamento e salvataggio LIAR (Benchmark)
    print("\nCaricamento LIAR Benchmark...")
    df_liar = load_liar(spark, LIAR_BASE)
    save_parquet(df_liar, OUT_LIAR)

    # 4. Caricamento e salvataggio USE24-XD (2024)
    print("\nCaricamento USE24-XD 2024...")
    df_use24 = load_use24(spark, USE24_PATH)
    save_parquet(df_use24, OUT_USE24)

    # 5. Verifica integrita' dei Parquet prodotti
    print("\nVerifica integrita' Parquet...")
    verify_parquets(spark, {
        "PHEME (Legacy 2016)":     OUT_PHEME,
        "LIAR (Benchmark)":        OUT_LIAR,
        "USE24-XD (Moderno 2024)": OUT_USE24,
    })

    spark.stop()
    print("Pipeline ETL completata.")
