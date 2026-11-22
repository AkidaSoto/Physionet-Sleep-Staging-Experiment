"""Unit tests for Experiment 75 (UnifiedBiGRUStagingNet, features, causality, calibration)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_staging_bigru_unified as unified  # noqa: E402
from physionet_sleep.algorithms.eeg_tracks import FS  # noqa: E402
from physionet_sleep.experiments.bigru_staging import (  # noqa: E402
    DEFAULT_EPOCH_DIM,
    DEFAULT_STEP_DIM,
    NIGHT_HISTORY_SOURCE_COLUMNS,
    UnifiedBiGRUStagingNet,
    block_epoch_mean,
    calibrate_prior_power,
    compute_night_history,
    compute_sigma_ratio,
    compute_split_support,
    extract_occipital_theta,
    fit_prior_power_oof,
)
from physionet_sleep.experiments.complementary_n2_events import _recent_support  # noqa: E402
from physionet_sleep.experiments.gated_staging import StagingTerms, stage_loss  # noqa: E402


def test_model_initialization_shapes_and_zero_init_epoch_stem() -> None:
    torch.manual_seed(7)
    model = UnifiedBiGRUStagingNet(step_dim=105, epoch_dim=15, width=32, dropout=0.15).eval()
    x = torch.randn(2, 12, 120)
    recorded = torch.ones(2, 12, dtype=torch.bool)
    recorded[1, 8:] = False
    x[1, 8:] = float("nan")

    x_epoch_perturbed = x.clone()
    x_epoch_perturbed[:, :, 105:] += 5.0

    with torch.inference_mode():
        terms = model(x, recorded, return_components=True)
        logits_perturbed = model(x_epoch_perturbed, recorded)

    assert isinstance(terms, StagingTerms)
    assert terms.logits.shape == (2, 12, 5)
    assert terms.past_gate.shape == (2, 12, 5)
    assert terms.future_gate.shape == (2, 12, 5)
    assert torch.isfinite(terms.logits).all()

    # Zero-init epoch projection and zero-init RNN heads keep init logits equal to local.
    torch.testing.assert_close(terms.logits, logits_perturbed, rtol=0, atol=1e-6)
    torch.testing.assert_close(terms.logits, terms.local, rtol=0, atol=1e-6)
    assert (terms.past == 0).all()
    assert (terms.future == 0).all()
    torch.testing.assert_close(terms.past_gate, torch.full_like(terms.past_gate, 0.8), rtol=0, atol=1e-5)

    # Padded positions stay zero.
    assert (terms.logits[1, 8:] == 0).all()
    assert (terms.local[1, 8:] == 0).all()


def test_model_causality_anti_causality_and_padding_invariance() -> None:
    torch.manual_seed(11)
    model = UnifiedBiGRUStagingNet(step_dim=20, epoch_dim=4, width=24, dropout=0.0)
    # Randomize RNN heads so past/future branches are non-zero.
    for head in (model.past_classifier, model.future_classifier):
        nn_weight = head[-1].weight
        torch.nn.init.normal_(nn_weight, std=0.2)
    model.eval()

    x_short = torch.randn(1, 9, 24)
    rec_short = torch.ones(1, 9, dtype=torch.bool)
    x_long = torch.randn(1, 15, 24)
    rec_long = torch.ones(1, 15, dtype=torch.bool)

    x_batch = torch.zeros(2, 15, 24)
    x_batch[0, :9] = x_short[0]
    x_batch[1] = x_long[0]
    rec_batch = torch.zeros(2, 15, dtype=torch.bool)
    rec_batch[0, :9] = True
    rec_batch[1] = True

    with torch.inference_mode():
        t_single = model(x_short, rec_short, return_components=True)
        t_batch = model(x_batch, rec_batch, return_components=True)

        # Invariance to trailing batch padding across every term.
        for field in t_single._fields:
            torch.testing.assert_close(
                getattr(t_single, field)[0, :9],
                getattr(t_batch, field)[0, :9],
                rtol=0,
                atol=1e-5,
            )

        # Boundary availability masks.
        assert (t_single.past[0, 0] == 0).all()
        assert (t_single.future[0, 8] == 0).all()
        assert not (t_single.past[0, 1:] == 0).all()
        assert not (t_single.future[0, :8] == 0).all()
        torch.testing.assert_close(
            t_single.logits, t_single.local + t_single.past + t_single.future
        )

        # Strict past causality: perturbing t > 4 cannot change local[:, :5] or past[:, :5].
        x_future_changed = x_short.clone()
        x_future_changed[:, 5:] += 3.0
        t_fc = model(x_future_changed, rec_short, return_components=True)
        torch.testing.assert_close(t_fc.local[:, :5], t_single.local[:, :5], rtol=0, atol=1e-6)
        torch.testing.assert_close(t_fc.past[:, :5], t_single.past[:, :5], rtol=0, atol=1e-6)

        # Strict future anti-causality: perturbing t < 4 cannot change local[:, 4:] or future[:, 4:].
        x_past_changed = x_short.clone()
        x_past_changed[:, :4] -= 3.0
        t_pc = model(x_past_changed, rec_short, return_components=True)
        torch.testing.assert_close(t_pc.local[:, 4:], t_single.local[:, 4:], rtol=0, atol=1e-6)
        torch.testing.assert_close(t_pc.future[:, 4:], t_single.future[:, 4:], rtol=0, atol=1e-6)


def test_model_gradients_reach_all_branches() -> None:
    torch.manual_seed(3)
    model = UnifiedBiGRUStagingNet(step_dim=10, epoch_dim=3, width=16, dropout=0.0).train()
    x = torch.randn(2, 10, 13)
    recorded = torch.ones(2, 10, dtype=torch.bool)
    recorded[1, 7:] = False
    y = torch.randint(0, 5, (2, 10))
    y[1, 7:] = -1
    weights = torch.ones(5)

    out = model(x, recorded, return_components=True)
    loss = stage_loss(out.logits, y, weights) + 0.25 * stage_loss(out.local, y, weights)
    loss.backward()

    assert model.step_proj.weight.grad is not None
    assert model.epoch_proj.weight.grad is not None
    assert model.past_rnn.weight_ih_l0.grad is not None
    assert model.future_rnn.weight_ih_l0.grad is not None
    assert model.gates.weight.grad is not None
    assert (model.epoch_proj.weight.grad.abs() > 0).any()


def test_occipital_theta_sigma_ratio_and_epoch_mean() -> None:
    n_sec = 45
    t = np.arange(n_sec * FS, dtype=np.float32) / FS
    theta_wave = 20.0 * np.sin(2.0 * np.pi * 6.0 * t)
    df = extract_occipital_theta(theta_wave, theta_wave)
    assert list(df.columns) == ["second", "o_rel_theta"]
    assert len(df) == n_sec
    assert df["o_rel_theta"].iloc[2:-2].mean() > 0.90

    # Verify NaN handling and block_epoch_mean over partial trailing epoch.
    values = df["o_rel_theta"].to_numpy().copy()
    values[:30] = 0.5
    values[30:] = np.nan
    epoch_means = block_epoch_mean(values, n_epochs=2)
    assert epoch_means.shape == (2,)
    np.testing.assert_allclose(epoch_means[0], 0.5, atol=1e-6)
    assert np.isnan(epoch_means[1])

    tracks = pd.DataFrame(
        {"c_rel_sigma": [0.3, 0.0], "c_rel_alpha": [0.2, 0.0], "c_rel_beta": [0.1, 0.0]}
    )
    ratio = compute_sigma_ratio(tracks)
    np.testing.assert_allclose(ratio, [0.3 / (0.3 + 1e-6), 0.0], rtol=1e-5)


def test_split_support_and_night_history() -> None:
    kc = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    spindle = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rec_k, rec_sp = compute_split_support(kc, spindle)
    assert np.isnan(rec_k[0]) and np.isnan(rec_sp[0])
    np.testing.assert_allclose(rec_k[2], 2.0 ** (-0.5), atol=1e-6)
    np.testing.assert_allclose(rec_sp[2], 0.0, atol=1e-6)
    np.testing.assert_allclose(rec_sp[3], 2.0 ** (-0.5), atol=1e-6)
    pooled = np.fmax(rec_k, rec_sp)
    expected = _recent_support(kc, spindle)
    known = np.isfinite(expected)
    np.testing.assert_allclose(pooled[known], expected[known], atol=1e-6)

    n_seconds = 300
    n_slots = n_seconds // 6
    rows6 = pd.DataFrame(
        {
            "slot": np.arange(n_slots),
            NIGHT_HISTORY_SOURCE_COLUMNS[0]: np.linspace(0.0, 0.8, n_slots),
            NIGHT_HISTORY_SOURCE_COLUMNS[1]: np.linspace(-10.0, -2.0, n_slots),
            NIGHT_HISTORY_SOURCE_COLUMNS[2]: np.linspace(-5.0, 5.0, n_slots),
        }
    )
    hist = compute_night_history(rows6, n_seconds)
    assert hist.shape == (10, 7)
    assert np.isfinite(hist).all()
    np.testing.assert_allclose(hist[:, 2], np.linspace(0.0, 1.0, 10), atol=1e-6)


def test_prepare_recordings_handles_absent_occipital_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "window_cache"
    dev_marker_dir = tmp_path / "dev_markers"
    (cache_dir / "tracks1s").mkdir(parents=True)
    (cache_dir / "rows6").mkdir(parents=True)
    dev_marker_dir.mkdir(parents=True)

    rng = np.random.default_rng(7)
    subjects = ["s_fit", "s_val"]
    rows = []
    for s in subjects:
        n_sec = 180
        n_ep = n_sec // 30
        n_slots = n_sec // 6
        track_df = pd.DataFrame({"second": np.arange(n_sec, dtype=np.int64)})
        for col in (*unified.BASE_STEP_TRACKS, "c_rel_sigma"):
            track_df[col] = rng.uniform(0.05, 0.4, size=n_sec).astype(np.float32)
        track_df.to_parquet(cache_dir / "tracks1s" / f"{s}.parquet", index=False)

        row_cols = {
            "slot": np.arange(n_slots, dtype=np.int64),
        }
        for col in unified.SEQUENCE.FEATURES:
            if col != "spindle_bursts":
                row_cols[col] = rng.normal(size=n_slots).astype(np.float32)
        for col in NIGHT_HISTORY_SOURCE_COLUMNS:
            row_cols[col] = rng.uniform(0.0, 1.0, size=n_slots).astype(np.float32)
        pd.DataFrame(row_cols).to_parquet(cache_dir / "rows6" / f"{s}.parquet", index=False)

        markers = pd.DataFrame(
            rng.uniform(0.0, 1.0, size=(n_ep, len(unified.expanded.FEATURES))),
            index=pd.RangeIndex(n_ep, name="epoch_index"),
            columns=list(unified.expanded.FEATURES),
        )
        markers.to_parquet(dev_marker_dir / f"{s}.parquet")
        for ep in range(n_ep):
            rows.append({"subject_id": s, "epoch_index": ep, unified.LABEL: ep % 5})

    frame = pd.DataFrame(rows)
    monkeypatch.setattr(unified, "CACHE", cache_dir)
    monkeypatch.setattr(unified, "DEV_MARKER_CACHE", dev_marker_dir)

    recordings, scaling = unified.prepare_recordings(
        frame,
        {"s_fit"},
        lambda: None,
        occipital_dir=tmp_path / "nonexistent_occipital",
        allow_missing_occipital=True,
    )
    assert set(recordings) == {"s_fit", "s_val"}
    for rec in recordings.values():
        assert rec.features.shape == (6, DEFAULT_STEP_DIM + DEFAULT_EPOCH_DIM)
        assert torch.isfinite(rec.features).all()
    assert scaling["step_finite_counts"][scaling["step_signals"].index("o_rel_theta")] == 0


def test_holdout_disjointness_guard() -> None:
    reserved = set((unified.RAW / "selected_records_holdout.txt").read_text().split())
    sample_holdout = next(iter(sorted(reserved)))
    with pytest.raises(ValueError, match="Holdout leakage"):
        unified.assert_holdout_disjoint({sample_holdout})


def test_fit_bigru_exact_resume_and_early_stopping(tmp_path: Path) -> None:
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    recordings = {}
    fit_ids = [f"tr_{i}" for i in range(6)]
    val_ids = ["va_0", "va_1"]
    for idx, s in enumerate(fit_ids + val_ids):
        n_ep = 12 + (idx % 3)
        feats = torch.from_numpy(
            rng.normal(size=(n_ep, DEFAULT_STEP_DIM + DEFAULT_EPOCH_DIM)).astype(np.float32)
        )
        labels = torch.tensor([i % 5 for i in range(n_ep)], dtype=torch.long)
        recordings[s] = unified.SEQUENCE.Recording(feats, labels)

    cfg = unified.TrainConfig(width=16, max_passes=2, min_passes=6, patience=5, batch_size=4)
    pause = tmp_path / "PAUSE"
    paused, state = unified.fit_bigru(
        recordings,
        fit_ids,
        val_ids,
        seed=7,
        directory=tmp_path / "resume",
        pause_path=pause,
        deadline=float("inf"),
        config=cfg,
        smoke_pause_updates=1,
    )
    assert paused is None
    assert state["next_pass"] == 0 and state["next_batch"] == 1

    resumed, summary = unified.fit_bigru(
        recordings,
        fit_ids,
        val_ids,
        seed=7,
        directory=tmp_path / "resume",
        pause_path=pause,
        deadline=float("inf"),
        config=cfg,
    )
    control, _ = unified.fit_bigru(
        recordings,
        fit_ids,
        val_ids,
        seed=7,
        directory=tmp_path / "control",
        pause_path=pause,
        deadline=float("inf"),
        config=cfg,
    )
    assert resumed is not None and control is not None
    assert summary["optimizer_updates"] == 4
    for name, tensor in resumed.state_dict().items():
        torch.testing.assert_close(tensor, control.state_dict()[name], rtol=0, atol=0)

    # Verify early-stopping predicate: never before min_passes=6, stops after patience=5.
    assert not unified.stopped({"next_pass": 6, "best_pass": 1}, unified.TrainConfig())
    assert unified.stopped({"next_pass": 7, "best_pass": 2}, unified.TrainConfig())
    assert not unified.stopped({"next_pass": 10, "best_pass": 6}, unified.TrainConfig())
    assert unified.stopped({"next_pass": 11, "best_pass": 6}, unified.TrainConfig())


def test_oof_prior_power_calibration() -> None:
    rng = np.random.default_rng(42)
    prior = np.array([0.2, 0.05, 0.50, 0.10, 0.15])
    y = rng.choice(5, size=800, p=prior)
    logits = np.log(prior)[None, :] + rng.normal(scale=0.6, size=(len(y), 5))
    logits[np.arange(len(y)), y] += 1.6
    logits -= logits.max(axis=1, keepdims=True)
    true_post = np.exp(logits)
    true_post /= true_post.sum(axis=1, keepdims=True)
    # Strongly over-weighted toward rare classes so positive exponent helps macro-F1.
    raw = calibrate_prior_power(true_post, prior, -1.0)
    folds = np.repeat(np.arange(5), len(y) // 5)

    cal, details = fit_prior_power_oof(raw, y, folds)
    assert len(details) == 5
    np.testing.assert_allclose(cal.sum(axis=1), 1.0, atol=1e-5)
    assert all(row["exponent"] > 0.0 for row in details)

    # Modifying held-out fold 0 cannot change fold 0's fitted exponent or prior.
    raw_mod = raw.copy()
    raw_mod[folds == 0] = 0.2
    _, details_mod = fit_prior_power_oof(raw_mod, y, folds)
    assert details_mod[0] == details[0]
