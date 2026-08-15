from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from physionet_sleep.diagnostics.metrics import rank_auc


@dataclass(slots=True)
class StageFitScore:
    score: float
    auc: float
    model_kind: str
    n_samples: int
    n_features: int
    used_features: list[str]
    fallback_feature_auc: list[float] | None = None


def score_stage_fit(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: list[str],
    target_class,
    contrast_classes: list | None = None,
    minimum_samples: int = 10,
    include_interactions: bool = True,
) -> StageFitScore:
    labels = np.asarray(labels)
    X = np.asarray(X, dtype=float)

    if contrast_classes is None:
        contrast_classes = [c for c in np.unique(labels).tolist() if c != target_class]

    keep = np.isin(labels, [target_class, *contrast_classes])
    if int(np.sum(keep)) < minimum_samples:
        return StageFitScore(0.0, 0.5, "insufficient_samples", int(np.sum(keep)), 0, [])

    y = labels[keep]
    X_sub = X[keep]
    y_bin = (y == target_class).astype(int)
    n0 = int(np.sum(y_bin == 0))
    n1 = int(np.sum(y_bin == 1))
    if min(n0, n1) < minimum_samples:
        return StageFitScore(0.0, 0.5, "insufficient_class_balance", int(y_bin.size), 0, [])

    valid_cols = []
    for i in range(X_sub.shape[1]):
        col = X_sub[:, i]
        if int(np.sum(np.isfinite(col))) < minimum_samples:
            continue
        if np.nanstd(col) <= 0:
            continue
        valid_cols.append(i)
    if not valid_cols:
        return StageFitScore(0.0, 0.5, "no_valid_features", int(y_bin.size), 0, [])

    X_use = X_sub[:, valid_cols]
    used_features = [feature_names[i] for i in valid_cols]
    total = n0 + n1
    sample_weight = np.zeros(y_bin.shape[0], dtype=float)
    sample_weight[y_bin == 0] = total / (2.0 * n0)
    sample_weight[y_bin == 1] = total / (2.0 * n1)

    try:
        if include_interactions:
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                StandardScaler(),
                LogisticRegression(max_iter=2000, solver="liblinear"),
            )
            model_kind = "logreg_interactions"
        else:
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=2000, solver="liblinear"),
            )
            model_kind = "logreg_linear"
        model.fit(X_use, y_bin, logisticregression__sample_weight=sample_weight)
        pred = model.predict_proba(X_use)[:, 1]
        auc = float(roc_auc_score(y_bin, pred, sample_weight=sample_weight))
        return StageFitScore(
            score=abs(auc - 0.5),
            auc=auc,
            model_kind=model_kind,
            n_samples=int(y_bin.size),
            n_features=len(valid_cols),
            used_features=used_features,
        )
    except Exception:
        aucs = []
        for i in range(X_use.shape[1]):
            pos = X_use[y_bin == 1, i]
            neg = X_use[y_bin == 0, i]
            auc = rank_auc(pos, neg)
            aucs.append(float(auc) if np.isfinite(auc) else float("nan"))
        finite = [abs(auc - 0.5) for auc in aucs if np.isfinite(auc)]
        best = max(finite) if finite else 0.0
        return StageFitScore(
            score=float(best),
            auc=float(best + 0.5),
            model_kind="feature_auc_fallback",
            n_samples=int(y_bin.size),
            n_features=len(valid_cols),
            used_features=used_features,
            fallback_feature_auc=aucs,
        )
