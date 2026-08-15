from __future__ import annotations

import numpy as np


def binary_confusion_counts(y_true, y_pred) -> dict[str, int]:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    n = min(truth.size, pred.size)
    truth = truth[:n]
    pred = pred[:n]
    return {
        "tp": int(np.sum(pred & truth)),
        "fp": int(np.sum(pred & ~truth)),
        "fn": int(np.sum(~pred & truth)),
        "tn": int(np.sum(~pred & ~truth)),
    }


def precision_recall_f1(*, tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def specificity_score(*, tn: int, fp: int) -> float:
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def confusion_matrix_from_arrays(y_true, y_pred, *, labels: list | None = None) -> tuple[np.ndarray, list]:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = min(yt.size, yp.size)
    yt = yt[:n]
    yp = yp[:n]
    if labels is None:
        labels = _ordered_union(yt, yp)
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b in zip(yt.tolist(), yp.tolist(), strict=False):
        if a in index and b in index:
            cm[index[a], index[b]] += 1
    return cm, labels


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    scores = []
    for i in range(cm.shape[0]):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        scores.append(precision_recall_f1(tp=tp, fp=fp, fn=fn)["f1"])
    return float(np.mean(scores)) if scores else float("nan")


def cohen_kappa_from_confusion(cm: np.ndarray) -> float:
    total = int(cm.sum())
    if total == 0:
        return float("nan")
    po = float(np.trace(cm) / total)
    row_m = cm.sum(axis=1) / total
    col_m = cm.sum(axis=0) / total
    pe = float(np.sum(row_m * col_m))
    if pe >= 1.0:
        return 0.0
    return float((po - pe) / (1.0 - pe))


def multiclass_classification_summary(y_true, y_pred, *, labels: list | None = None) -> dict[str, object]:
    cm, labels = confusion_matrix_from_arrays(y_true, y_pred, labels=labels)
    per_class = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        metrics = precision_recall_f1(tp=tp, fp=fp, fn=fn)
        per_class.append(
            {
                "label": label,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "support": int(cm[i, :].sum()),
            }
        )
    return {
        "confusion": cm,
        "labels": labels,
        "macro_f1": macro_f1_from_confusion(cm),
        "kappa": cohen_kappa_from_confusion(cm),
        "per_class": per_class,
        "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan"),
    }


def _ordered_union(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    seen = []
    for value in y_true.tolist() + y_pred.tolist():
        if _is_missing(value):
            continue
        if value not in seen:
            seen.append(value)
    return seen


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False
