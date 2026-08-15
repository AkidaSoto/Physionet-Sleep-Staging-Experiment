from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from physionet_sleep.core.types import DatasetRun, DiagnosticResult
from physionet_sleep.diagnostics.base import BaseDiagnostic
from physionet_sleep.diagnostics.metrics import cohens_d, rank_auc
from physionet_sleep.diagnostics.series import iter_harvested_series


@dataclass(slots=True)
class PairwiseFeatureDiagnostic(BaseDiagnostic):
    name: str = "pairwise_features"
    track_name: str = "stage_seconds"
    minimum_class_samples: int = 10

    def run(self, dataset_run: DatasetRun) -> DiagnosticResult:
        labels = dataset_run.alignment.tracks.get(self.track_name)
        if labels is None:
            return DiagnosticResult(name=self.name, metadata={"status": "skipped", "reason": "missing labels"})
        labels = np.asarray(labels).astype(int)
        class_ids = sorted(int(v) for v in np.unique(labels[np.isfinite(labels)]))
        if len(class_ids) < 2:
            return DiagnosticResult(name=self.name, metadata={"status": "skipped", "reason": "need >=2 classes"})

        harvested = list(iter_harvested_series(dataset_run, target_track=self.track_name))
        pair_ids = list(combinations(class_ids, 2))
        d_matrix = np.full((len(harvested), len(pair_ids)), np.nan, dtype=float)
        auc_matrix = np.full((len(harvested), len(pair_ids)), np.nan, dtype=float)

        for i, series in enumerate(harvested):
            vals = np.asarray(series.values, dtype=float)
            ok = np.isfinite(vals) & np.isfinite(labels)
            vals = vals[ok]
            yy = labels[ok]
            for j, (ca, cb) in enumerate(pair_ids):
                xa = vals[yy == ca]
                xb = vals[yy == cb]
                if xa.size < self.minimum_class_samples or xb.size < self.minimum_class_samples:
                    continue
                d_matrix[i, j] = abs(cohens_d(xa, xb))
                auc = rank_auc(xa, xb)
                auc_matrix[i, j] = max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan

        pair_labels = [f"{a} vs {b}" for a, b in pair_ids]
        stage_sep_d = _pair_matrix_from_feature_matrix(d_matrix, pair_ids, class_ids, agg="l2")
        stage_sep_auc = _pair_matrix_from_feature_matrix(np.abs(auc_matrix - 0.5), pair_ids, class_ids, agg="l2")
        mean_d = _pair_matrix_from_feature_matrix(d_matrix, pair_ids, class_ids, agg="mean")
        mean_auc = _pair_matrix_from_feature_matrix(auc_matrix, pair_ids, class_ids, agg="mean")

        return DiagnosticResult(
            name=self.name,
            summaries={
                "feature_names": [s.name for s in harvested],
                "class_ids": class_ids,
                "pair_ids": pair_ids,
                "pair_labels": pair_labels,
                "d_matrix": d_matrix,
                "auc_matrix": auc_matrix,
                "stage_sep_l2_d": stage_sep_d,
                "stage_sep_l2_auc": stage_sep_auc,
                "stage_sep_mean_d": mean_d,
                "stage_sep_mean_auc": mean_auc,
            },
            metadata={"track_name": self.track_name, "series_count": len(harvested)},
        )


def _pair_matrix_from_feature_matrix(feature_pair_matrix: np.ndarray, pair_ids: list[tuple[int, int]], class_ids: list[int], *, agg: str) -> np.ndarray:
    n = len(class_ids)
    out = np.full((n, n), np.nan, dtype=float)
    class_index = {c: i for i, c in enumerate(class_ids)}
    for col, (ca, cb) in enumerate(pair_ids):
        values = feature_pair_matrix[:, col]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        if agg == "l2":
            score = float(np.sqrt(np.sum(finite**2)))
        else:
            score = float(np.mean(finite))
        ia = class_index[ca]
        ib = class_index[cb]
        out[ia, ib] = score
        out[ib, ia] = score
    np.fill_diagonal(out, 0.0)
    return out
