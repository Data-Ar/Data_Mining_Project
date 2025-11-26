from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_all_metrics(y_true: List[int], y_prob: List[float]) -> Dict[str, float]:
    """Return standard binary classification metrics."""

    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
    }


def format_results(results: Dict[str, Dict[str, float]]) -> str:
    """Pretty-print helper used by CLI scripts."""

    lines = []
    for group, metrics in results.items():
        lines.append(f"\n====== {group.upper()} ======")
        if metrics is None:
            lines.append("No samples in this group.")
            continue
        lines.append(f"AUC:         {metrics['auc']:.4f}")
        lines.append(f"Precision:   {metrics['precision']:.4f}")
        lines.append(f"Recall:      {metrics['recall']:.4f}")
        lines.append(f"Specificity: {metrics['specificity']:.4f}")
        lines.append(f"F1 Score:    {metrics['f1']:.4f}")
    return "\n".join(lines)


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print overall / female / male metrics (for compatibility with original notebook)."""
    print(format_results(results))

