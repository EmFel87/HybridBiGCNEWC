# =============================================================================
# FILE:    models/ewc.py
# SCOPO:   Logica di Continual Learning tramite Elastic Weight
#          Consolidation (EWC) per il progetto "Fake News Detection:
#          2016 vs 2024". Contiene esclusivamente le funzioni per il
#          calcolo della Fisher Information Matrix (FIM) diagonale sul
#          task storico (PHEME) e per il calcolo della penale
#          quadratica EWC applicata durante il fine-tuning sul task
#          recente (USE24).
#          Questo modulo e' agnostico rispetto all'architettura: opera
#          su qualsiasi nn.Module tramite named_parameters(), e non
#          contiene alcun riferimento specifico a HybridGatedBiGCN.
#          Nessuna logica di training loop, valutazione o gestione dei
#          seed risiede in questo modulo: e' responsabilita' esclusiva
#          di run_hybrid.py, che importa queste funzioni.
# DIPENDENZE: torch
# MODULO:  Continual Learning — Elastic Weight Consolidation (EWC)
# =============================================================================

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader


def compute_fisher_matrix(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Calcola la Fisher Information Matrix diagonale sul task storico.

    La Fisher Information Matrix (FIM) misura la curvatura della loss
    rispetto a ciascun parametro del modello: parametri con FIM elevata
    sono "importanti" per il task corrente (nel progetto, PHEME 2016) e
    verranno vincolati dalla penale EWC (``compute_ewc_penalty``)
    durante il fine-tuning su un task successivo (USE24 2024).

    Approssimazione diagonale adottata (Kirkpatrick et al., 2017,
    "Overcoming catastrophic forgetting in neural networks"):

    .. math::
        F_i \\approx \\mathbb{E}\\left[
            \\left( \\frac{\\partial}{\\partial \\theta_i}
            \\log p(y \\mid x, \\theta) \\right)^2
        \\right]

    In pratica, per un modello con loss binaria (``BCEWithLogitsLoss``),
    :math:`F_i` viene stimata come media del quadrato dei gradienti
    della loss rispetto a ciascun parametro :math:`\\theta_i`, calcolata
    su tutti i mini-batch del ``loader`` fornito. L'accumulo e'
    normalizzato dividendo per il numero totale di batch, cosi' che
    ``fisher_dict`` rappresenti una media e non una somma.

    La funzione restituisce anche il clone dei pesi ottimali
    :math:`\\theta^*_A` al momento della chiamata (tipicamente il
    modello appena addestrato sul task storico, prima di qualsiasi
    fine-tuning): questi valori costituiscono il punto di riferimento
    attorno al quale la penale EWC penalizzera' i futuri scostamenti.

    Nota implementativa: il modello viene impostato in modalita'
    ``eval()`` (per disattivare il Dropout durante il calcolo dei
    gradienti "di riferimento"), ma i gradienti restano attivi tramite
    ``loss.backward()`` — a differenza della valutazione standard, qui
    non si usa ``torch.no_grad()`` perche' i gradienti sono l'oggetto
    stesso del calcolo.

    Args:
        model: Modello PyTorch (tipicamente ``HybridGatedBiGCN``) gia'
            addestrato sul task storico. I suoi pesi correnti sono
            trattati come :math:`\\theta^*_A`.
        loader: DataLoader (PyTorch Geometric) del training set del
            task storico su cui calcolare la FIM.
        criterion: Funzione di loss (tipicamente
            ``BCEWithLogitsLoss``), identica a quella usata nel
            training originale.
        device: Dispositivo di calcolo su cui spostare i batch.

    Returns:
        Tupla ``(fisher_dict, opt_params)`` dove:
          ``fisher_dict``: dizionario ``{nome_parametro: tensore FIM
            diagonale}``, stessa forma del parametro corrispondente.
          ``opt_params``: dizionario ``{nome_parametro: clone dei pesi
            ottimali theta*}``, salvato prima di qualsiasi
            aggiornamento successivo.
    """
    fisher_dict: dict[str, torch.Tensor] = {}
    opt_params:  dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            opt_params[name]  = param.data.clone()
            fisher_dict[name] = torch.zeros_like(param.data)

    model.eval()
    n_batches = len(loader)

    for batch_data in loader:
        batch_data = batch_data.to(device)
        model.zero_grad()

        loss = criterion(
            model(batch_data).view(-1), batch_data.y.view(-1)
        )
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Accumulo normalizzato: media sul numero di batch,
                # non somma, cosi' che fisher_dict sia comparabile
                # tra dataset di dimensioni diverse.
                fisher_dict[name] += (
                    param.grad.data.pow(2) / n_batches
                )

    return fisher_dict, opt_params


def compute_ewc_penalty(
    model:       nn.Module,
    fisher_dict: dict[str, torch.Tensor],
    opt_params:  dict[str, torch.Tensor],
    lambda_val:  float,
) -> torch.Tensor:
    """Calcola la penale EWC completa da sommare alla loss del nuovo task.

    Implementa la formula di regolarizzazione EWC (Kirkpatrick et al.,
    2017), incluso il coefficiente di scala:

    .. math::
        \\mathcal{L}_{EWC} = \\frac{\\lambda}{2} \\sum_i F_i
            \\left( \\theta_i - \\theta^*_{A,i} \\right)^2

    dove :math:`F_i` e' la Fisher Information Matrix diagonale
    calcolata da ``compute_fisher_matrix`` sul task storico (PHEME),
    :math:`\\theta^*_{A,i}` sono i pesi ottimali dello stesso task, e
    :math:`\\theta_i` sono i pesi correnti del modello durante il
    fine-tuning sul nuovo task (USE24). Il coefficiente
    :math:`\\lambda` (``lambda_val``) bilancia la protezione della
    memoria storica rispetto alla velocita' di adattamento al nuovo
    dominio: valori piu' alti proteggono piu' fortemente
    :math:`\\theta^*_A` a scapito della plasticita' sul nuovo task.

    Il valore restituito da questa funzione e' gia' scalato per
    :math:`\\lambda / 2`: il chiamante deve sommarlo direttamente alla
    task loss senza applicare ulteriori fattori di scala, ad esempio:

        ``loss = criterion(out, y) + compute_ewc_penalty(model,
        fisher_dict, opt_params, lambda_val)``

    Args:
        model: Modello in fase di fine-tuning sul nuovo task. I suoi
            parametri correnti (:math:`\\theta`) sono confrontati con
            ``opt_params``.
        fisher_dict: FIM diagonale prodotta da
            ``compute_fisher_matrix`` sul task storico.
        opt_params: Pesi ottimali :math:`\\theta^*_A` prodotti da
            ``compute_fisher_matrix`` sul task storico.
        lambda_val: Coefficiente di bilanciamento :math:`\\lambda`
            della penale EWC.

    Returns:
        Tensore scalare Float32 con la penale EWC gia' scalata per
        :math:`\\lambda / 2`, pronta per essere sommata alla task
        loss.
    """
    penalty = torch.tensor(
        0.0, device=next(model.parameters()).device
    )
    for name, param in model.named_parameters():
        if param.requires_grad and name in fisher_dict:
            fisher    = fisher_dict[name]
            opt_param = opt_params[name]
            penalty  += (fisher * (param - opt_param).pow(2)).sum()

    return (lambda_val / 2.0) * penalty

def compute_fisher_group_stats(
    fisher_dict: dict[str, torch.Tensor],
    param_masks: dict[str, list[str]]
) -> dict[str, dict[str, float]]:
    """Estrae media, mediana e numero di parametri della FIM raggruppati per macro-componente."""
    stats = {}
    for group_name, prefixes in param_masks.items():
        g_vals = []
        for n, tensor in fisher_dict.items():
            if any(pref in n for pref in prefixes):
                g_vals.append(tensor.flatten())
        if g_vals:
            g_vals = torch.cat(g_vals)
            stats[group_name] = {
                "mean": float(g_vals.mean().item()),
                "median": float(g_vals.median().item()),
                "num": int(g_vals.numel())
            }
    return stats
