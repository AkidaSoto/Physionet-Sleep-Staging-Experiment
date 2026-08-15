from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physionet_sleep.core.types import DatasetRun
from physionet_sleep.diagnostics.metrics import rank_auc
from physionet_sleep.diagnostics.series import iter_harvested_series


@dataclass(slots=True)
class SubjectQCReport:
    subject_ids: list[str]
    class_ids: list[int]
    class_distribution: np.ndarray
    class_distribution_z: np.ndarray
    subject_sep_score: np.ndarray
    subject_sep_z: np.ndarray
    zscore_by_class: np.ndarray
    zscore_by_feature: np.ndarray
    feature_names: list[str]


def compute_subject_qc(runs: list[DatasetRun], *, track_name: str = "stage_seconds") -> SubjectQCReport:
    subject_ids = [str(run.metadata.get("record_id", i)) for i, run in enumerate(runs)]
    all_classes = sorted(
        {
            int(v)
            for run in runs
            for v in np.unique(np.asarray(run.alignment.tracks.get(track_name, []))[np.isfinite(np.asarray(run.alignment.tracks.get(track_name, [])))])
        }
    )
    class_index = {c: i for i, c in enumerate(all_classes)}
    class_dist = np.full((len(runs), len(all_classes)), np.nan, dtype=float)

    series_names: list[str] = []
    per_subject_class_scores: list[np.ndarray] = []
    per_subject_feature_scores: list[dict[str, float]] = []

    for i, run in enumerate(runs):
        labels = run.alignment.tracks.get(track_name)
        if labels is None:
            per_subject_class_scores.append(np.full(len(all_classes), np.nan))
            per_subject_feature_scores.append({})
            continue
        labels = np.asarray(labels).astype(int)
        ok_lab = np.isfinite(labels)
        if ok_lab.any():
            yy = labels[ok_lab]
            for cls in all_classes:
                class_dist[i, class_index[cls]] = float(np.mean(yy == cls))

        harvested = list(iter_harvested_series(run, target_track=track_name))
        feature_scores: dict[str, float] = {}
        class_scores = np.full(len(all_classes), np.nan, dtype=float)
        class_buf: list[list[float]] = [[] for _ in all_classes]
        for series in harvested:
            vals = np.asarray(series.values, dtype=float)
            ok = np.isfinite(vals) & np.isfinite(labels)
            vals = vals[ok]
            yy = labels[ok]
            aucs = []
            for cls in all_classes:
                pos = vals[yy == cls]
                neg = vals[yy != cls]
                if pos.size < 5 or neg.size < 5:
                    continue
                auc = rank_auc(pos, neg)
                auc_sep = max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan
                if np.isfinite(auc_sep):
                    aucs.append(auc_sep)
                    class_buf[class_index[cls]].append(auc_sep)
            if aucs:
                feature_scores[series.name] = float(np.sqrt(np.sum(np.square(aucs))))
        for cls_i, vals in enumerate(class_buf):
            if vals:
                class_scores[cls_i] = float(np.mean(vals))
        per_subject_class_scores.append(class_scores)
        per_subject_feature_scores.append(feature_scores)
        for name in feature_scores:
            if name not in series_names:
                series_names.append(name)

    by_class = np.vstack(per_subject_class_scores) if per_subject_class_scores else np.empty((0, len(all_classes)))
    by_feature = np.full((len(runs), len(series_names)), np.nan, dtype=float)
    for i, scores in enumerate(per_subject_feature_scores):
        for j, name in enumerate(series_names):
            if name in scores:
                by_feature[i, j] = scores[name]

    subject_sep_score = np.nanmean(by_class, axis=1) if by_class.size else np.array([])
    return SubjectQCReport(
        subject_ids=subject_ids,
        class_ids=all_classes,
        class_distribution=class_dist,
        class_distribution_z=_zscore_cols(class_dist),
        subject_sep_score=subject_sep_score,
        subject_sep_z=_zscore_1d(subject_sep_score),
        zscore_by_class=_zscore_cols(by_class),
        zscore_by_feature=_zscore_cols(by_feature),
        feature_names=series_names,
    )


def _zscore_cols(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    if x.ndim != 2:
        return out
    for j in range(x.shape[1]):
        col = x[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        mu = np.mean(col[ok])
        sd = np.std(col[ok])
        if sd <= 0:
            continue
        out[ok, j] = (col[ok] - mu) / sd
    return out


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return out
    mu = np.mean(x[ok])
    sd = np.std(x[ok])
    if sd <= 0:
        return out
    out[ok] = (x[ok] - mu) / sd
    return out
