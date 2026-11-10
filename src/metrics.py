"""Classification metrics and operating-threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true, y_proba, threshold: float = 0.5
) -> dict:
    """Return a complete metric bundle for a set of probabilities."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "n_samples": int(len(y_true)),
    }


def threshold_table(
    y_true, y_proba, start: float = 0.05, stop: float = 0.95, step: float = 0.005
) -> pd.DataFrame:
    """Precision/recall/F1 for a grid of decision thresholds."""
    rows = []
    for threshold in np.round(np.arange(start, stop + step / 2, step), 3):
        y_pred = (np.asarray(y_proba) >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def find_best_threshold(
    y_true,
    y_proba,
    min_precision: float | None = None,
    min_recall: float | None = None,
    objective: str = "f1",
) -> tuple[float, pd.DataFrame]:
    """Choose an operating threshold on out-of-fold predictions.

    Parameters
    ----------
    objective:
        ``"f1"`` maximises F1 (ties broken toward recall). ``"recall"``
        maximises recall subject to the optional precision floor, which is the
        right objective for a retention campaign where missed churners are
        expensive but outreach capacity is finite.
    """
    table = threshold_table(y_true, y_proba)

    candidates = table
    if min_precision is not None:
        filtered = table[table["precision"] >= min_precision]
        if not filtered.empty:
            candidates = filtered
    if min_recall is not None:
        filtered = candidates[candidates["recall"] >= min_recall]
        if not filtered.empty:
            candidates = filtered

    if objective == "recall":
        if min_recall is not None:
            # Among cut-offs that hit the recall SLA, keep the most precise
            # (i.e. the least wasteful) one.
            best = candidates.sort_values(
                ["precision", "recall"], ascending=[False, False]
            ).iloc[0]
        else:
            best = candidates.sort_values(
                ["recall", "precision"], ascending=[False, False]
            ).iloc[0]
        return float(best["threshold"]), table

    best_f1 = candidates["f1"].max()
    finalists = candidates[np.isclose(candidates["f1"], best_f1)]
    best = finalists.sort_values(
        ["recall", "precision"], ascending=[False, False]
    ).iloc[0]
    return float(best["threshold"]), table

