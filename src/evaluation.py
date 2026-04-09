"""Evaluation metrics for link prediction and ranking tasks.

Reference:
    Hanley, J.A. & McNeil, B.J. "The meaning and use of the area under
    a receiver operating characteristic (ROC) curve."
    Radiology 143(1), 29–36 (1982).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_metrics(scores, labels):
    """Compute AUROC, AUPRC, MRR, and Hits@10.

    Parameters
    ----------
    scores : array-like of float
        Predicted scores (higher = more likely positive).
    labels : array-like of int
        Binary labels (1 = positive, 0 = negative).

    Returns
    -------
    dict with keys: auroc, auprc, mrr, hits@10
    """
    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=int)

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    # MRR and Hits@10: for each positive, find its rank among all scores
    pos_scores = scores[labels == 1]
    reciprocal_ranks, hits = [], []
    for ps in pos_scores:
        rank = int(np.sum(scores >= ps))      # 1-indexed; ties broken conservatively
        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
        hits.append(1 if rank <= 10 else 0)

    return {
        'auroc':   float(auroc),
        'auprc':   float(auprc),
        'mrr':     float(np.mean(reciprocal_ranks)),
        'hits@10': float(np.mean(hits)),
    }


# ── Bootstrap confidence intervals (Cell 97) ──────────────────────────────────

def bootstrap_auroc_ci(scores_arr, labels_arr, n_boot=100, rng_=None):
    """Percentile bootstrap 95% CI for AUROC.

    Returns (point_estimate, ci_lo, ci_hi). Returns (nan, nan, nan) if there
    are fewer than 2 positives or 2 negatives, or if fewer than 10 valid
    bootstrap samples are generated.

    Parameters
    ----------
    scores_arr : array-like of float
        Prediction scores.
    labels_arr : array-like of int
        Binary labels (1 = positive, 0 = negative).
    n_boot : int
        Number of bootstrap samples.
    rng_ : np.random.RandomState, optional
        RNG for reproducibility. If None, uses np.random.

    Returns
    -------
    tuple of (float, float, float)
        (point_estimate, ci_lo, ci_hi) or (nan, nan, nan) if insufficient data.
    """
    if rng_ is None:
        rng_ = np.random
    n = len(scores_arr)
    if labels_arr.sum() < 2 or (1 - labels_arr).sum() < 2:
        return np.nan, np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        idx = rng_.randint(0, n, n)
        if labels_arr[idx].sum() == 0 or labels_arr[idx].sum() == n:
            continue
        try:
            boots.append(roc_auc_score(labels_arr[idx], scores_arr[idx]))
        except Exception:
            continue
    if len(boots) < 10:
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(boots)), float(lo), float(hi)
