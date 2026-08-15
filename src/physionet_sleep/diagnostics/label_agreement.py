from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from physionet_sleep.core.types import DatasetRun
from physionet_sleep.evaluation import (
    cohen_kappa_from_confusion,
    confusion_matrix_from_arrays,
    macro_f1_from_confusion,
)


@dataclass(slots=True)
class LabelAgreementPair:
    source_a: str
    source_b: str
    confusion: np.ndarray
    macro_f1: float
    kappa: float


def compute_label_agreement(
    dataset_run: DatasetRun,
    *,
    track_names: list[str],
) -> list[LabelAgreementPair]:
    pairs: list[LabelAgreementPair] = []
    available = {name: np.asarray(dataset_run.alignment.tracks[name]) for name in track_names if name in dataset_run.alignment.tracks}
    for a, b in combinations(available.keys(), 2):
        y1 = np.asarray(available[a]).astype(int)
        y2 = np.asarray(available[b]).astype(int)
        n = min(y1.size, y2.size)
        y1 = y1[:n]
        y2 = y2[:n]
        ok = np.isfinite(y1) & np.isfinite(y2)
        y1 = y1[ok]
        y2 = y2[ok]
        if y1.size == 0:
            continue
        classes = sorted(set(y1.tolist()) | set(y2.tolist()))
        cm, _ = confusion_matrix_from_arrays(y1, y2, labels=classes)
        pairs.append(
            LabelAgreementPair(
                source_a=a,
                source_b=b,
                confusion=cm,
                macro_f1=macro_f1_from_confusion(cm),
                kappa=cohen_kappa_from_confusion(cm),
            )
        )
    return pairs
