"""Unit tests for the Experiment 76 multi-scale tempered HGB pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from physionet_sleep.experiments.multiscale_hgb import (
    BASE_SIGNALS,
    CHIN_COL,
    EYE_COL,
    MARKER_COLUMNS,
    NIGHT_HISTORY_COLUMNS,
    OCCIPITAL_COLUMN,
    SEQUENCE_FEATURES,
    SEQUENCE_TRACKS,
    SLOW_COL,
    CacheLayout,
    build_fit_matrix,
    calibrate_oof_prior_power,
    compute_night_history,
    compute_tempered_weights,
    encode_subject_features,
    feature_column_names,
    fit_fold_moments,
    fit_tempered_hgb,
    has_occipital_cache,
    load_subject_night,
    score_subjects_streaming,
)


def _write_synthetic_subject(
    root: Path,
    subject: str,
    n_epochs: int,
    rng: np.random.Generator,
    *,
    is_added: bool = False,
    with_occipital: bool = False,
    stage_shift: bool = True,
) -> pd.DataFrame:
    wdir = root / ("added_cache" if is_added else "dev_cache")
    mdir = root / ("added_markers" if is_added else "dev_markers")
    (wdir / "tracks1s").mkdir(parents=True, exist_ok=True)
    (wdir / "rows6").mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    n_seconds = n_epochs * 30
    stages = np.tile(np.arange(5, dtype=np.int64), n_epochs // 5 + 1)[:n_epochs]

    track_cols = list(dict.fromkeys([*SEQUENCE_TRACKS, "c_rel_sigma", "c_rel_alpha", "c_rel_beta"]))
    tracks = pd.DataFrame(rng.uniform(0.1, 1.0, (n_seconds, len(track_cols))), columns=track_cols)
    if stage_shift:
        tracks["c_log_power"] += np.repeat(stages * 0.6, 30)
        tracks["c_rel_sigma"] += np.repeat((stages == 2).astype(np.float32) * 1.2, 30)
    tracks.to_parquet(wdir / "tracks1s" / f"{subject}.parquet", index=False)

    row_cols = [c for c in SEQUENCE_FEATURES if c != "spindle_bursts"]
    n_rows6 = n_epochs * 5
    rows = pd.DataFrame(rng.normal(size=(n_rows6, len(row_cols))), columns=row_cols)
    rows["slot"] = np.arange(n_rows6, dtype=np.int64)
    rows[SLOW_COL] = np.clip(rng.uniform(0.0, 1.0, n_rows6), 0.0, 1.0)
    if stage_shift:
        rows[CHIN_COL] = rng.normal(size=n_rows6) - np.repeat((stages == 4).astype(np.float32) * 2.0, 5)
    rows.to_parquet(wdir / "rows6" / f"{subject}.parquet", index=False)

    markers = pd.DataFrame(
        rng.normal(size=(n_epochs, len(MARKER_COLUMNS))).astype(np.float32),
        columns=list(MARKER_COLUMNS),
        index=pd.RangeIndex(n_epochs, name="epoch_index"),
    )
    markers.to_parquet(mdir / f"{subject}.parquet")

    if with_occipital:
        odir = root / "occipital_cache"
        odir.mkdir(parents=True, exist_ok=True)
        occ = pd.DataFrame(
            {OCCIPITAL_COLUMN: rng.normal(size=n_epochs).astype(np.float32)},
            index=pd.RangeIndex(n_epochs, name="epoch_index"),
        )
        occ.to_parquet(odir / f"{subject}.parquet")

    return pd.DataFrame(
        {
            "subject_id": subject,
            "epoch_index": np.arange(n_epochs, dtype=np.int64),
            "label.stage_30s": stages,
        }
    )


def _layout(root: Path, added_subjects: set[str], use_occ: bool = False) -> CacheLayout:
    return CacheLayout(
        dev_window_cache=root / "dev_cache",
        added_window_cache=root / "added_cache",
        dev_marker_cache=root / "dev_markers",
        added_marker_cache=root / "added_markers",
        added_subjects=frozenset(added_subjects),
        occipital_cache=(root / "occipital_cache") if use_occ else None,
    )


def test_feature_column_names_and_occipital_toggle(tmp_path):
    cols = feature_column_names(include_occipital=False)
    cols_occ = feature_column_names(include_occipital=True)
    assert len(cols) == 306
    assert len(cols_occ) == 309
    assert len(set(cols_occ)) == 309
    assert cols_occ[-3:] == [f"{OCCIPITAL_COLUMN}@ep-1", f"{OCCIPITAL_COLUMN}@ep+0", f"{OCCIPITAL_COLUMN}@ep+1"]

    assert not has_occipital_cache(tmp_path / "missing", ["s1"])
    assert not has_occipital_cache(None, ["s1"])

    rng = np.random.default_rng(7)
    _write_synthetic_subject(tmp_path, "s1", 15, rng, with_occipital=True)
    assert has_occipital_cache(tmp_path / "occipital_cache", ["s1"])
    assert not has_occipital_cache(tmp_path / "occipital_cache", ["s1", "s2"])


def test_streaming_encoding_boundaries_and_moments_isolation(tmp_path):
    rng = np.random.default_rng(11)
    f1 = _write_synthetic_subject(tmp_path, "fit-1", 30, rng, is_added=False)
    f2 = _write_synthetic_subject(tmp_path, "fit-2", 30, rng, is_added=True)
    ev = _write_synthetic_subject(tmp_path, "eval-1", 30, rng, is_added=False)

    # Corrupt eval-1's c_log_power scale wildly; fitting moments must NOT see eval-1!
    eval_tracks_path = tmp_path / "dev_cache" / "tracks1s" / "eval-1.parquet"
    eval_tracks = pd.read_parquet(eval_tracks_path)
    eval_tracks["c_log_power"] = 9999.0
    eval_tracks.to_parquet(eval_tracks_path, index=False)

    layout = _layout(tmp_path, {"fit-2"})
    moments = fit_fold_moments(layout, ["fit-1", "fit-2"], include_occipital=False)
    fit_df = pd.concat([f1, f2], ignore_index=True)
    X_fit, y_fit = build_fit_matrix(layout, fit_df, moments)

    assert X_fit.dtype == np.float32
    assert X_fit.shape == (len(fit_df), 306)
    assert np.isfinite(X_fit).all()
    # Fitting moments should have small mean (~0) and reasonable SD, unaffected by eval-1's 9999.0
    c_log_idx = BASE_SIGNALS.index("c_log_power")
    assert abs(float(moments.signal_mean[c_log_idx])) < 5.0

    # Verify boundary zero-padding = 0.0 (fold training mean) at epoch 0 for negative context offsets.
    dense, markers, history, occ = load_subject_night(layout, "fit-1", include_occipital=False)
    X_ep0 = encode_subject_features(dense, markers, history, occ, np.array([0]), moments)
    cols = feature_column_names(include_occipital=False)
    neg_step_cols = [i for i, c in enumerate(cols) if c.endswith(("@-30s", "@-24s", "@-18s", "@-12s", "@-6s", "@ep-1", "@ep-2"))]
    # Wait: @ep-1 includes marker@ep-1 and non_fast@ep-1 which are all 0.0 at epoch 0!
    assert np.allclose(X_ep0[0, neg_step_cols], 0.0)

    # Verify positive boundary padding at final epoch (epoch 29).
    X_ep_last = encode_subject_features(dense, markers, history, occ, np.array([29]), moments)
    pos_cols = [i for i, c in enumerate(cols) if c.endswith(("@+30s", "@+36s", "@+42s", "@+48s", "@+54s", "@ep+1", "@ep+2", "@ep[+11..+20]", "@ep[+6..+10]", "@ep[+3..+5]"))]
    assert np.allclose(X_ep_last[0, pos_cols], 0.0)

    # Verify out-of-range epoch index fails fast.
    with pytest.raises(ValueError, match="recording length"):
        encode_subject_features(dense, markers, history, occ, np.array([30]), moments)


def test_compute_night_history_contract():
    n_epochs = 20
    rows = pd.DataFrame(
        {
            "slot": np.arange(n_epochs * 5, dtype=np.int64),
            SLOW_COL: np.linspace(0.0, 1.0, n_epochs * 5),
            CHIN_COL: np.concatenate([np.ones(50), -np.ones(30), -2.0 * np.ones(20)]),
            EYE_COL: np.concatenate([np.zeros(50), np.ones(30), 3.0 * np.ones(20)]),
        }
    )
    hist = compute_night_history(rows, n_epochs * 30)
    assert hist.shape == (n_epochs, len(NIGHT_HISTORY_COLUMNS))
    assert hist.dtype == np.float32
    assert np.isfinite(hist).all()
    assert hist[0, 0] == pytest.approx(0.0)
    assert hist[-1, 2] == pytest.approx(1.0)
    # Near the end of the night, low chin + high EOG anchors should raise atonia & nearby_eye_atonia
    assert hist[14, 5] > hist[2, 5]
    assert hist[14, 6] > hist[2, 6]


def test_tempered_weights_and_end_to_end_pipeline(tmp_path):
    rng = np.random.default_rng(42)
    frames = []
    added = set()
    for i in range(6):
        subj = f"subj-{i}"
        is_add = i >= 4
        if is_add:
            added.add(subj)
        frames.append(_write_synthetic_subject(tmp_path, subj, 40, rng, is_added=is_add, with_occipital=True))

    layout = _layout(tmp_path, added, use_occ=True)
    all_df = pd.concat(frames, ignore_index=True)

    # 2-fold test with shuffled evaluation order
    oof_probs = []
    oof_labels = []
    oof_folds = []
    fold_priors = []
    for f, eval_ids in enumerate((["subj-0", "subj-1"], ["subj-2", "subj-3"])):
        fit_df = all_df[~all_df.subject_id.isin(eval_ids)].reset_index(drop=True)
        eval_df = all_df[all_df.subject_id.isin(eval_ids)].sample(frac=1.0, random_state=f).reset_index(drop=True)

        moments = fit_fold_moments(layout, fit_df.subject_id.unique(), include_occipital=True)
        X_fit, y_fit = build_fit_matrix(layout, fit_df, moments)
        assert X_fit.shape == (len(fit_df), 309)

        sw, counts, class_weights = compute_tempered_weights(y_fit, n_classes=5, weight_power=0.50)
        assert sw.mean() == pytest.approx(1.0, abs=1e-6)
        assert np.allclose(class_weights, counts.astype(np.float64) ** (-0.50))

        clf = fit_tempered_hgb(X_fit, y_fit, sw, seed=7 + f, max_iter=15)
        probs = score_subjects_streaming(clf, layout, eval_df, moments)
        assert probs.shape == (len(eval_df), 5)
        assert probs.dtype == np.float32
        assert np.isfinite(probs).all()
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

        oof_probs.append(probs)
        oof_labels.append(eval_df["label.stage_30s"].to_numpy(dtype=np.int64))
        oof_folds.append(np.full(len(eval_df), f, dtype=np.int64))
        fold_priors.append(counts / counts.sum())

    cal_probs, cal_info = calibrate_oof_prior_power(
        np.vstack(oof_probs),
        np.concatenate(oof_labels),
        np.concatenate(oof_folds),
        np.asarray(fold_priors),
    )
    assert cal_probs.shape == (len(np.concatenate(oof_labels)), 5)
    assert np.isfinite(cal_probs).all()
    assert np.allclose(cal_probs.sum(axis=1), 1.0, atol=1e-5)
    assert set(cal_info["lofo_alphas"].keys()) == {"0", "1"}

    # Missing stage raises ValueError
    with pytest.raises(ValueError, match="Every stage must occur"):
        compute_tempered_weights(np.array([0, 1, 2, 3]), n_classes=5, weight_power=0.5)
