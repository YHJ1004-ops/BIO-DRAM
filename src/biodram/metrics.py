from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix


def compute_metrics(y_true, y_prob, num_classes: int) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        out["macro_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        out["macro_auc"] = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    recalls = []
    for i in range(num_classes):
        denom = cm[i, :].sum()
        recalls.append(float(cm[i, i] / denom) if denom > 0 else 0.0)
    out.update({f"recall_class_{i}": r for i, r in enumerate(recalls)})
    return out
