from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((x.size - 1) * vx + (y.size - 1) * vy) / (x.size + y.size - 2)
    if pooled <= 0:
        return 0.0
    return float((x.mean() - y.mean()) / np.sqrt(pooled))


def rank_auc(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    x_pos = np.asarray(x_pos, dtype=float)
    x_neg = np.asarray(x_neg, dtype=float)
    x_pos = x_pos[np.isfinite(x_pos)]
    x_neg = x_neg[np.isfinite(x_neg)]
    if x_pos.size == 0 or x_neg.size == 0:
        return float("nan")
    joined = np.concatenate([x_pos, x_neg])
    ranks = rankdata(joined, method="average")
    r_pos = ranks[: x_pos.size].sum()
    u_pos = r_pos - x_pos.size * (x_pos.size + 1) / 2.0
    return float(u_pos / (x_pos.size * x_neg.size))
