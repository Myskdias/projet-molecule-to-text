"""
eval_metrics.py
================

Évaluation officielle ALTEGRAD pour Molecular Graph Captioning.

MÉTRIQUES :
-----------
- BLEU-4 (avec smoothing)
- BERTScore F1 (RoBERTa-base)

Ce fichier est volontairement SIMPLE et TRANSPARENT :
- aucune dépendance au reste du pipeline
- fonctionne avec n'importe quelles listes (preds, refs)

UTILISATION TYPIQUE :
--------------------
from eval_metrics import evaluate_all

bleu4, bert_f1 = evaluate_all(preds, refs)
"""

from typing import List, Tuple

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bert_score


# ============================================================
# BLEU-4
# ============================================================

def compute_bleu4(preds: List[str], refs: List[str]) -> float:
    """
    Calcule le BLEU-4 corpus-level (avec smoothing).

    Args:
        preds : List[str]
            Textes prédits
        refs : List[str]
            Textes de référence (ground truth)

    Returns:
        bleu4 : float
    """
    assert len(preds) == len(refs), "Preds and refs must have same length"

    smoothing = SmoothingFunction().method4

    refs_bleu = [[r.split()] for r in refs]
    preds_bleu = [p.split() for p in preds]

    bleu4 = corpus_bleu(
        refs_bleu,
        preds_bleu,
        smoothing_function=smoothing
    )

    return bleu4


# ============================================================
# BERTScore (RoBERTa-base)
# ============================================================

def compute_bertscore(
    preds: List[str],
    refs: List[str],
    device: str = None
) -> float:
    """
    Calcule le BERTScore F1 moyen (RoBERTa-base).

    Args:
        preds : List[str]
            Textes prédits
        refs : List[str]
            Textes de référence
        device : str, optional
            "cuda" ou "cpu" (auto si None)

    Returns:
        bert_f1 : float
    """
    assert len(preds) == len(refs), "Preds and refs must have same length"

    P, R, F1 = bert_score(
        preds,
        refs,
        model_type="roberta-base",
        lang="en",
        device=device,
        verbose=True
    )

    return F1.mean().item()


# ============================================================
# ÉVALUATION COMPLÈTE (officielle ALTEGRAD)
# ============================================================

def evaluate_all(
    preds: List[str],
    refs: List[str],
    device: str = None
) -> Tuple[float, float]:
    """
    Évalue un modèle avec les métriques officielles ALTEGRAD.

    Returns:
        bleu4 : float
        bert_f1 : float
    """
    print("\n[Evaluation] Computing BLEU-4...")
    bleu4 = compute_bleu4(preds, refs)
    print(f"[Evaluation] BLEU-4: {bleu4:.4f}")

    print("\n[Evaluation] Computing BERTScore (RoBERTa-base)...")
    bert_f1 = compute_bertscore(preds, refs, device=device)
    print(f"[Evaluation] BERTScore F1: {bert_f1:.4f}")

    return bleu4, bert_f1


# ============================================================
# CLI / DEBUG
# ============================================================

if __name__ == "__main__":
    # Exemple minimal pour sanity check
    preds = [
        "This molecule contains a benzene ring and a hydroxyl group.",
        "The compound is an aliphatic alcohol with one oxygen atom."
    ]

    refs = [
        "This molecule has a benzene ring and a hydroxyl functional group.",
        "The molecule is an aliphatic alcohol containing one oxygen."
    ]

    evaluate_all(preds, refs)
