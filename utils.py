# =============================================================================
# FILE:    utils.py
# SCOPO:   Utility condivise tra tutti i moduli sperimentali del progetto
#          "Fake News Detection: 2016 vs 2024" (baseline semantica,
#          baseline topologica, modello ibrido + EWC). Centralizza:
#            - impostazione dei seed per la riproducibilita' totale
#            - dispositivo di calcolo (CPU/GPU)
#            - calcolo delle metriche di valutazione (accuracy, precision,
#              recall, F1 per classe, F1 macro, confusion matrix)
#            - strutture dati per l'aggregazione dei risultati multi-seed
#            - reportistica statistica finale (media ± deviazione standard)
#            - salvataggio di checkpoint dei pesi e di grafici
#          Nessuna logica di modello o di training risiede in questo
#          modulo: qui vive esclusivamente il codice trasversale che i
#          moduli di training (train_*.py) e le architetture (models/)
#          importano senza duplicazione.
# DIPENDENZE: torch, numpy, scikit-learn, matplotlib
# MODULO:  Utility di base — nessuna dipendenza da altri moduli del progetto
# =============================================================================


# ##############################################################################
# ### FASE 0: IMPORT E COSTANTI GLOBALI                                      ###
# ##############################################################################

import os
import random
from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt

# -- Dispositivo di calcolo condiviso da tutti i moduli -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Nomi delle metriche raccolte per ciascuna run multi-seed -----------------
#    Condivisi da tutti gli esperimenti (baseline semantica, topologica,
#    ibrido): garantiscono che print_stats() e i dizionari di risultati
#    abbiano sempre la stessa struttura, indipendentemente dal modello.
CORE_METRIC_KEYS: list[str] = [
    "acc",
    "p_fake", "p_real",
    "r_fake", "r_real",
    "f1_fake", "f1_real",
    "f1_macro",
]


# ##############################################################################
# ### FASE 1: RIPRODUCIBILITA' — GESTIONE DEI SEED                          ###
# ##############################################################################

def set_seed(seed: int) -> None:
    """Fissa tutti i generatori di numeri casuali per la riproducibilita' totale.

    Imposta il seed su tre livelli distinti che intervengono nella pipeline
    sperimentale: il modulo random di Python (usato indirettamente da
    alcune utility di libreria), NumPy (operazioni su array e sampling)
    e PyTorch (inizializzazione dei pesi, dropout, shuffle dei DataLoader).
    Su GPU viene inoltre fissato il seed di tutti i device CUDA disponibili
    tramite manual_seed_all, necessario per la riproducibilita' quando il
    training viene eseguito su un ambiente multi-GPU.

    Questa funzione deve essere invocata all'inizio di ogni run del
    protocollo multi-seed, prima di qualsiasi split dei dati o
    inizializzazione del modello: gli split stratificati di
    scikit-learn usano random_state=seed indipendentemente, ma
    l'inizializzazione dei pesi della rete e l'ordine di shuffle dei
    DataLoader dipendono esclusivamente da questa funzione.

    Args:
        seed: Valore intero del seed da propagare a tutti i generatori.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ##############################################################################
# ### FASE 2: CALCOLO DELLE METRICHE DI VALUTAZIONE                         ###
# ##############################################################################

def compute_classification_metrics(
    all_labels: list[float],
    all_preds:  list[float],
) -> dict[str, Any]:
    """Calcola l'intero pacchetto di metriche di classificazione binaria.

    Centralizza la logica di valutazione condivisa da tutti gli esperimenti
    (baseline semantica, baseline topologica, modello ibrido): accuracy,
    precision/recall/F1 per singola classe, F1 macro e confusion matrix.

    La convenzione di etichettatura e' fissa in tutto il progetto:
      classe 0 = Fake/Rumor
      classe 1 = Real/Non-Rumor
    Il parametro labels=[0, 1] passato a precision_recall_fscore_support
    garantisce un ordine di output deterministico indipendente dalla
    distribuzione delle classi nel batch valutato, e zero_division=0
    evita eccezioni quando una classe e' assente dalle predizioni
    (es. un modello collassato che predice sempre la stessa classe).

    Args:
        all_labels: Lista di label vere (0/1), tipicamente accumulata
                    durante l'iterazione su un DataLoader di test.
        all_preds:  Lista di predizioni binarie (0/1) del modello,
                    gia' sogliate a 0.5 sulla probabilita' sigmoidale.

    Returns:
        Dizionario con le chiavi:
          acc, p_fake, p_real, r_fake, r_real,
          f1_fake, f1_real, f1_macro, cm
        dove cm e' la confusion matrix 2x2 di scikit-learn (array NumPy)
        e tutte le altre chiavi sono float scalari.
    """
    acc = accuracy_score(all_labels, all_preds)

    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], zero_division=0
    )

    p_fake, p_real   = precision[0], precision[1]
    r_fake, r_real   = recall[0], recall[1]
    f1_fake, f1_real = f1_per_class[0], f1_per_class[1]
    f1_macro         = float(np.mean(f1_per_class))

    cm = confusion_matrix(all_labels, all_preds)

    return {
        "acc":      acc,
        "p_fake":   p_fake,
        "p_real":   p_real,
        "r_fake":   r_fake,
        "r_real":   r_real,
        "f1_fake":  f1_fake,
        "f1_real":  f1_real,
        "f1_macro": f1_macro,
        "cm":       cm,
    }


# ##############################################################################
# ### FASE 3: AGGREGAZIONE DEI RISULTATI MULTI-SEED                         ###
# ##############################################################################

def init_results_dict(
    include_loss_history: bool = False,
) -> dict[str, list]:
    """Inizializza il dizionario di accumulo dei risultati multi-seed.

    Ogni chiave in CORE_METRIC_KEYS diventa una lista vuota che verra'
    popolata con un valore per ciascun seed del protocollo multi-run
    (append_run_results). La chiave "cm" accumula le confusion matrix
    complete per un'eventuale analisi aggregata (es. somma delle matrici
    su tutti i seed).

    Il parametro include_loss_history abilita due chiavi aggiuntive
    ("train_loss_history", "val_loss_history"), usate esclusivamente
    dal dataset su cui il modello viene effettivamente addestrato
    (es. PHEME nella baseline semantica); i dataset di solo test
    (es. USE24 nel test a freddo) non necessitano di questa cronologia.

    Args:
        include_loss_history: Se True, aggiunge le chiavi per la
                              cronologia delle loss di training/validation.

    Returns:
        Dizionario {nome_metrica: lista_vuota} pronto per essere
        popolato da append_run_results su ciascun seed.
    """
    results: dict[str, list] = {
        key: [] for key in CORE_METRIC_KEYS
    }
    results["cm"] = []

    if include_loss_history:
        results["train_loss_history"] = []
        results["val_loss_history"]   = []

    return results


def append_run_results(
    results_dict: dict[str, list],
    metrics:      dict[str, Any],
) -> None:
    """Accoda le metriche di una singola run al dizionario di accumulo.

    Effettua l'append in-place di ciascuna metrica calcolata da
    compute_classification_metrics() nella lista corrispondente del
    dizionario di accumulo multi-seed, mantenendo l'ordine di
    inserimento allineato all'ordine dei seed nel ciclo esterno.

    Args:
        results_dict: Dizionario prodotto da init_results_dict(), che
                      viene modificato in-place.
        metrics:      Dizionario prodotto da compute_classification_metrics()
                      per la run corrente (un singolo seed).

    Returns:
        None. results_dict viene aggiornato in-place.
    """
    for key in CORE_METRIC_KEYS:
        results_dict[key].append(metrics[key])
    results_dict["cm"].append(metrics["cm"])


def print_stats(
    dataset_name: str,
    results_dict: dict[str, list],
) -> None:
    """Stampa media e deviazione standard delle metriche su tutti i seed.

    Formato di stampa uniforme per il confronto diretto tra esperimenti
    diversi (baseline semantica, topologica, ibrida) e tra dataset diversi
    (test set storico vs test di Concept Drift): ogni riga riporta il
    nome della metrica allineato a sinistra, seguito da media e
    deviazione standard su tutte le run del protocollo multi-seed.

    Args:
        dataset_name: Etichetta descrittiva del dataset/esperimento,
                      stampata come intestazione (es. "PHEME (Test
                      Set Storico)", "USE24 (Concept Drift a Freddo)").
        results_dict: Dizionario di accumulo popolato da
                      append_run_results() su tutti i seed del
                      protocollo multi-run.

    Returns:
        None. Stampa il report direttamente su stdout.
    """
    print(f"\n--- {dataset_name} ---")
    for metric in CORE_METRIC_KEYS:
        mean_val = np.mean(results_dict[metric])
        std_val  = np.std(results_dict[metric])
        print(f"{metric.ljust(10)} : {mean_val:.4f} ± {std_val:.4f}")


# ##############################################################################
# ### FASE 4: SALVATAGGIO CHECKPOINT E GRAFICI                              ###
# ##############################################################################

def save_checkpoint(
    model:       torch.nn.Module,
    output_path: str,
    seed:        Optional[int] = None,
) -> None:
    """Salva lo state_dict di un modello PyTorch su disco.

    Salva esclusivamente i pesi (state_dict) e non l'intero oggetto
    modello, seguendo la pratica raccomandata da PyTorch per la
    portabilita' tra versioni di libreria e per evitare la
    serializzazione di riferimenti a classi che potrebbero non essere
    disponibili in fase di caricamento (es. refactoring futuro del
    codice sorgente delle architetture).

    Se seed non e' None, viene incorporato nel nome del file per
    distinguere i checkpoint delle diverse run del protocollo
    multi-seed senza sovrascriverli a vicenda.

    Args:
        model:       Istanza di un modello PyTorch (nn.Module) da
                     salvare. Puo' trovarsi su CPU o GPU: lo
                     state_dict viene salvato cosi' com'e', il
                     caricamento successivo dovra' gestire la
                     rilocazione tramite map_location.
        output_path: Percorso di destinazione. Se seed e' specificato,
                     il suffisso "_seed{seed}" viene inserito prima
                     dell'estensione del file.
        seed:        Seed della run corrente, opzionale.

    Returns:
        None.
    """
    if seed is not None:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_seed{seed}{ext}"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Checkpoint salvato: {output_path}")


def plot_loss_curves(
    train_loss_history: list[float],
    val_loss_history:   list[float],
    output_path:         str,
    title:               str = "Training vs Validation Loss",
) -> None:
    """Genera e salva su disco il grafico delle curve di loss.

    Produce un grafico a linee con training loss e validation loss
    sovrapposte per epoca, utile per la diagnosi visiva di overfitting
    (divergenza tra le due curve) o di instabilita' del training
    (oscillazioni nella validation loss). Il grafico viene salvato
    come file immagine e la figura viene chiusa esplicitamente per
    evitare l'accumulo di figure aperte in memoria durante l'esecuzione
    del protocollo multi-seed.

    Args:
        train_loss_history: Lista di loss medie di training, una per
                            epoca.
        val_loss_history:   Lista di loss medie di validation, una per
                            epoca. Deve avere la stessa lunghezza di
                            train_loss_history.
        output_path:        Percorso del file immagine di destinazione
                            (es. "plots/pheme_seed42_loss.png").
        title:               Titolo del grafico.

    Returns:
        None.
    """
    epochs_range = range(1, len(train_loss_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_loss_history, label="Train Loss")
    plt.plot(epochs_range, val_loss_history,   label="Val Loss")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Grafico salvato: {output_path}")


if __name__ == "__main__":
    print("Modulo utils.py caricato.")
