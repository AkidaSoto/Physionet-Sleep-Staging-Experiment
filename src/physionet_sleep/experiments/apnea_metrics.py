from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physionet_sleep.core.types import DatasetRun, ExperimentResult
from physionet_sleep.evaluation import binary_confusion_counts, precision_recall_f1, specificity_score
from physionet_sleep.experiments.base import BaseExperiment


@dataclass(slots=True)
class ApneaSecondwiseExperiment(BaseExperiment):
    name: str = "apnea_secondwise"
    truth_track: str = "respiratory_event_truth"
    pred_track: str = "apnea_event_prediction_sec"

    def run(self, dataset_run: DatasetRun) -> ExperimentResult:
        truth = dataset_run.alignment.tracks.get(self.truth_track)
        pred = dataset_run.alignment.tracks.get(self.pred_track)
        if truth is None or pred is None:
            return ExperimentResult(
                name=self.name,
                metrics={},
                metadata={"status": "skipped", "reason": "missing prediction or truth track"},
            )

        truth = np.asarray(truth) > 0
        pred = np.asarray(pred) > 0
        n = min(truth.size, pred.size)
        truth = truth[:n]
        pred = pred[:n]
        counts = binary_confusion_counts(truth, pred)
        pr = precision_recall_f1(tp=counts["tp"], fp=counts["fp"], fn=counts["fn"])
        specificity = specificity_score(tn=counts["tn"], fp=counts["fp"])

        return ExperimentResult(
            name=self.name,
            metrics={
                "seconds_evaluated": n,
                **counts,
                **pr,
                "specificity": specificity,
            },
        )
