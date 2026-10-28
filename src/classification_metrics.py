"""Small dependency-free multiclass metrics for fixed class-index contracts."""
from __future__ import annotations

import numpy as np


def classification_metrics(y_true, y_pred, class_names):
    names = list(class_names)
    n = len(names)
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if truth.shape != pred.shape or truth.ndim != 1 or len(truth) == 0:
        raise ValueError("truth/pred must be nonempty equal-length vectors")
    if np.any((truth < 0) | (truth >= n) | (pred < 0) | (pred >= n)):
        raise ValueError("class index out of range")
    matrix = np.zeros((n, n), dtype=np.int64)
    np.add.at(matrix, (truth, pred), 1)
    per_class = []
    for i, name in enumerate(names):
        tp = int(matrix[i, i])
        support = int(matrix[i].sum())
        predicted = int(matrix[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"index": i, "class_name": name, "support": support,
                          "precision": precision, "recall": recall, "f1": f1})
    return {"n": len(truth), "accuracy": float(np.trace(matrix) / len(truth)),
            "macro_f1": float(np.mean([x["f1"] for x in per_class])),
            "balanced_accuracy": float(np.mean([x["recall"] for x in per_class])),
            "per_class": per_class, "confusion_matrix": matrix.tolist(),
            "matrix_orientation": "rows=true, columns=predicted"}
