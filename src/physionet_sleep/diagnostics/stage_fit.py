from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physionet_sleep.core.types import DatasetRun, DiagnosticResult
from physionet_sleep.diagnostics.base import BaseDiagnostic
from physionet_sleep.diagnostics.series import iter_harvested_series
from physionet_sleep.evaluation import score_stage_fit


@dataclass(slots=True)
class StageFitDiagnostic(BaseDiagnostic):
    name: str = "stage_fit"
    track_name: str = "stage_seconds"
    minimum_samples: int = 10
    include_interactions: bool = True

    def run(self, dataset_run: DatasetRun) -> DiagnosticResult:
        labels = dataset_run.alignment.tracks.get(self.track_name)
        if labels is None:
            return DiagnosticResult(
                name=self.name,
                summaries={"rows": []},
                metadata={"status": "skipped", "reason": f"missing track '{self.track_name}'"},
            )

        labels = np.asarray(labels)
        finite = np.isfinite(labels)
        if int(np.sum(finite)) < self.minimum_samples:
            return DiagnosticResult(
                name=self.name,
                summaries={"rows": []},
                metadata={"status": "skipped", "reason": "insufficient finite labels"},
            )

        harvested = list(iter_harvested_series(dataset_run, target_track=self.track_name))
        if not harvested:
            return DiagnosticResult(
                name=self.name,
                summaries={"rows": []},
                metadata={"status": "skipped", "reason": "no harvested series"},
            )

        feature_names = [series.name for series in harvested]
        X = np.column_stack([np.asarray(series.values, dtype=float) for series in harvested])[finite]
        y = labels[finite].astype(int)
        class_ids = sorted(int(v) for v in np.unique(y))
        class_map = (
            dataset_run.labels.tracks.get("stage_seconds_collapsed", None).class_map
            if "stage_seconds_collapsed" in dataset_run.labels.tracks
            else {}
        )

        rows = []
        for class_id in class_ids:
            fit = score_stage_fit(
                X,
                y,
                feature_names=feature_names,
                target_class=class_id,
                minimum_samples=self.minimum_samples,
                include_interactions=self.include_interactions,
            )
            rows.append(
                {
                    "class_id": class_id,
                    "class_label": class_map.get(class_id, str(class_id)),
                    "score": float(fit.score),
                    "auc": float(fit.auc),
                    "model_kind": fit.model_kind,
                    "n_samples": int(fit.n_samples),
                    "n_features": int(fit.n_features),
                    "used_features": fit.used_features,
                }
            )

        return DiagnosticResult(
            name=self.name,
            summaries={"rows": rows},
            metadata={
                "track_name": self.track_name,
                "class_ids": class_ids,
                "row_count": len(rows),
                "include_interactions": self.include_interactions,
            },
        )
