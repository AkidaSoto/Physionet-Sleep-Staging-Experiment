"""Feature reduction for the development-only Model C comparison."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from physionet_sleep.analysis.epoch_windows import DERIVED_FEATURES, SUBJ, TARGET, WindowEncoder, dense_signals
from physionet_sleep.experiments.tabular_runner import build_model


REMOVED_TRACKS = ("c_rel_sigma", "f_rel_delta")


def _dense_cached_night(
    cache_dir: Path,
    subject: str,
    tracks: list[str],
    features: list[str],
) -> np.ndarray:
    # Model C's spindle proxy depends on the raw sigma, alpha and beta tracks.
    track_columns = list(dict.fromkeys([*tracks, "c_rel_sigma", "c_rel_alpha", "c_rel_beta"]))
    row_columns = ["slot", *(name for name in features if name not in DERIVED_FEATURES)]
    track_frame = pd.read_parquet(cache_dir / "tracks1s" / f"{subject}.parquet", columns=track_columns)
    row_frame = pd.read_parquet(cache_dir / "rows6" / f"{subject}.parquet", columns=row_columns)
    return dense_signals(track_frame, row_frame, tracks, features)


def fit_feature_reduction_fold(
    cache_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    tracks: list[str],
    features: list[str],
    *,
    seed: int = 7,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit 18-signal HGB and return five-class probabilities in the input test row order.

    ``tracks`` and ``features`` are the unmodified Model C inputs. The caller
    supplies the original development split and combines results across folds.
    The matched HGB defaults include automatic early stopping, which draws a
    row-level validation subset entirely inside the outer training patients.
    """
    if len(tracks) + len(features) != 20 or not set(REMOVED_TRACKS).issubset(tracks):
        raise ValueError("feature reduction requires the original 20 Model C signals")
    if train.empty or test.empty:
        raise ValueError("a feature-reduction fold requires training and test epochs")
    train_subjects = sorted(train[SUBJ].astype(str).unique())
    if set(train_subjects).intersection(test[SUBJ].astype(str)):
        raise ValueError("feature-reduction training and test patients overlap")

    reduced_tracks = [name for name in tracks if name not in REMOVED_TRACKS]
    signal_names = [name.split(".")[-1] for name in reduced_tracks + features]
    encoder = WindowEncoder(signal_names, context=2)
    columns = encoder.column_names()

    start = perf_counter()
    dense = {
        subject: _dense_cached_night(cache_dir, subject, reduced_tracks, features)
        for subject in train_subjects
    }
    encoder.fit(dense, train_subjects)
    train_windows = encoder.transform(dense, train).reshape(len(train), len(columns))
    del dense
    train_preparation_s = perf_counter() - start

    model = build_model(model_family="hgb", task="classification", random_state=seed)
    start = perf_counter()
    model.fit(pd.DataFrame(train_windows, columns=columns), train[TARGET].to_numpy())
    fit_s = perf_counter() - start
    del train_windows

    # Process test patients one at a time to avoid holding a second fold-wide
    # window matrix or loading test data while the learner fits.
    probabilities = np.zeros((len(test), 5))
    classes = np.asarray(model.classes_, dtype=int)
    cache_window_s = predict_s = 0.0
    inference_start = perf_counter()
    for subject, positions in test.groupby(test[SUBJ].astype(str), sort=False).indices.items():
        start = perf_counter()
        night = _dense_cached_night(cache_dir, subject, reduced_tracks, features)
        windows = encoder.transform({subject: night}, test.iloc[positions]).reshape(len(positions), len(columns))
        model_frame = pd.DataFrame(windows, columns=columns)
        cache_window_s += perf_counter() - start

        start = perf_counter()
        probabilities[np.ix_(positions, classes)] = model.predict_proba(model_frame)
        predict_s += perf_counter() - start

    estimator = model.named_steps["model"]
    return probabilities, {
        "input_signals": len(signal_names),
        "input_columns": len(columns),
        "removed_tracks": list(REMOVED_TRACKS),
        "train_cache_to_windows_s": train_preparation_s,
        "fit_s": fit_s,
        "test_cache_to_windows_s": cache_window_s,
        "test_predict_s": predict_s,
        "test_cache_to_predictions_s": perf_counter() - inference_start,
        "boosting_iterations": int(estimator.n_iter_),
        "early_stopping_active": bool(estimator.do_early_stopping_),
        "early_stopping_validation": "library-default row split within the outer training patients",
        "raw_extraction_changed": False,
        "raw_extraction_timed": False,
        "raw_extraction_note": (
            "The shared recipe still computes frontal delta, and spindle_bursts still requires central sigma. "
            "Timing starts from the existing feature and track caches."
        ),
    }
