from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from physionet_sleep.core.types import DatasetRun, DiagnosticResult
from physionet_sleep.diagnostics.base import BaseDiagnostic
from physionet_sleep.diagnostics.metrics import cohens_d, rank_auc
from physionet_sleep.diagnostics.series import iter_harvested_series


@dataclass(slots=True)
class FeatureScreenDiagnostic(BaseDiagnostic):
    name: str = "feature_screen"
    track_name: str = "stage_seconds"
    minimum_class_samples: int = 10

    def run(self, dataset_run: DatasetRun) -> DiagnosticResult:
        labels = dataset_run.alignment.tracks.get(self.track_name)
        if labels is None:
            return DiagnosticResult(
                name=self.name,
                summaries={"rows": []},
                metadata={"status": "skipped", "reason": f"missing track '{self.track_name}'"},
            )

        labels = np.asarray(labels)
        feature_rows: list[dict[str, Any]] = []
        class_values = sorted(int(v) for v in np.unique(labels[np.isfinite(labels)]))
        class_map = (
            dataset_run.labels.tracks.get("stage_seconds_collapsed", None).class_map
            if "stage_seconds_collapsed" in dataset_run.labels.tracks
            else {}
        )

        for series in iter_harvested_series(dataset_run, target_track=self.track_name):
            vector = np.asarray(series.values, dtype=float)
            finite = np.isfinite(vector) & np.isfinite(labels)
            if finite.sum() < self.minimum_class_samples:
                continue
            vv = vector[finite]
            yy = labels[finite].astype(int)
            for class_id in class_values:
                pos = vv[yy == class_id]
                neg = vv[yy != class_id]
                if pos.size < self.minimum_class_samples or neg.size < self.minimum_class_samples:
                    continue
                d = cohens_d(pos, neg)
                auc = rank_auc(pos, neg)
                feature_rows.append(
                    {
                        "algorithm": series.algorithm,
                        "feature": series.name,
                        "source_kind": series.source_kind,
                        "class_id": class_id,
                        "class_label": class_map.get(class_id, str(class_id)),
                        "cohen_d": float(d),
                        "auc_sep": float(max(auc, 1.0 - auc)) if np.isfinite(auc) else float("nan"),
                        "n_pos": int(pos.size),
                        "n_neg": int(neg.size),
                    }
                )

        return DiagnosticResult(
            name=self.name,
            summaries={"rows": feature_rows},
            metadata={"track_name": self.track_name, "row_count": len(feature_rows)},
        )
