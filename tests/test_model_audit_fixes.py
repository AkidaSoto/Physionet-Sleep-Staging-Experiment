"""Tests for the model-audit fixes: time-based 6-s binning, the hgb model family, out-of-fold stage-1
probabilities for stage 2, and single class balancing in the TCN trainer."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from physionet_sleep.algorithms.feature_table import FeatureTableCompiler, _coerce_signal
from physionet_sleep.core.types import AlgorithmResult, SignalRecord, SignalSeries
from physionet_sleep.experiments import tabular_runner
from physionet_sleep.experiments.tabular_runner import build_model

REPO = Path(__file__).resolve().parents[1]
ROW = 6.0


def _load_stage2_script():
    path = REPO / "scripts" / "run_paper_matched_stage2_smoothers.py"
    spec = importlib.util.spec_from_file_location("run_paper_matched_stage2_smoothers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- 1. time-based binning -------------------------------------------------------------------------


def _frame_track(*, hop: float, start: float, duration: float, with_offset: bool = True) -> SignalSeries:
    """Spectrogram-like track whose value is its own frame-centre time."""
    times = np.arange(start, duration, hop)
    return SignalSeries(times.copy(), sample_rate=1.0 / hop, metadata={"start_seconds": start} if with_offset else {})


def test_frame_hop_that_does_not_divide_row_lands_on_true_rows():
    # 2-s STFT at 128 Hz with 90 % overlap: hop 26/128 s, first frame centred at 1 s (the spindle_features grid).
    duration = 7 * 3600.0
    n_rows = int(np.ceil(duration / ROW))
    series = _frame_track(hop=26 / 128, start=1.0, duration=duration)
    out = _coerce_signal(series, n_rows, ROW)
    row_start = np.arange(n_rows) * ROW
    # Every row's mean frame time must sit inside that row, all night long (index binning drifted ~6 min).
    assert np.all(out >= row_start) and np.all(out < row_start + ROW)
    assert np.isfinite(out).all()  # no NaN tail


def test_index_binning_would_have_drifted():
    # Guards the test above: the same track without the time axis was ~55 rows off by the end of the night.
    hop = 26 / 128
    rows_per_old_window = round(ROW / hop)
    drift_per_row = rows_per_old_window * hop - ROW
    assert drift_per_row * 4200 / ROW > 50


def test_late_first_frame_is_edge_filled_and_offset_respected():
    # 30-s EOG STFT at 64 Hz, 99 % overlap: hop 19/64 s, first frame centred at 15 s.
    duration = 3600.0
    n_rows = int(np.ceil(duration / ROW))
    series = _frame_track(hop=19 / 64, start=15.0, duration=duration - 15.0)
    out = _coerce_signal(series, n_rows, ROW)
    assert np.isfinite(out).all()
    # Rows 0 and 1 have no frame centre of their own and take row 2's value.
    assert out[0] == out[1] == out[2]
    inner = np.arange(3, n_rows - 3)
    assert np.all((out[inner] >= inner * ROW) & (out[inner] < inner * ROW + ROW))


def test_edge_fill_is_bounded():
    # A track that stops after 60 s must not be extended over the whole night.
    series = SignalSeries(np.ones(60), sample_rate=1.0)
    out = _coerce_signal(series, 100, ROW)
    assert np.isfinite(out[:15]).all()  # 60 s of data + 30 s of edge fill
    assert np.isnan(out[15:]).all()


def test_explicit_time_axis_is_used():
    times = np.array([0.5, 2.0, 6.5, 13.0, 13.5])
    series = SignalSeries(np.array([1.0, 3.0, 10.0, 20.0, 40.0]), sample_rate=1.0, metadata={"time_seconds": times})
    out = _coerce_signal(series, 3, ROW)
    np.testing.assert_allclose(out, [2.0, 10.0, 30.0])


def test_native_rate_signal_matches_block_mean():
    fs = 128.0
    values = np.random.default_rng(0).normal(size=int(fs * ROW * 50))
    out = _coerce_signal(SignalSeries(values, sample_rate=fs), 50, ROW)
    np.testing.assert_allclose(out, values.reshape(50, -1).mean(axis=1))


def test_binary_tracks_use_max_per_row():
    values = np.zeros(int(ROW * 4 * 10))
    values[int(ROW * 10) + 5] = 1.0  # one event in row 1
    out = _coerce_signal(SignalSeries(values, sample_rate=10.0, units="binary"), 4, ROW)
    np.testing.assert_array_equal(out, [0.0, 1.0, 0.0, 0.0])


def test_compiler_aligns_mixed_rate_tracks():
    duration = 600.0
    record = SignalRecord(channels={"eeg": SignalSeries(np.zeros(int(duration * 128)), sample_rate=128.0)})
    prior = {
        "spec": AlgorithmResult(
            name="spec",
            signals={"frame_time": _frame_track(hop=26 / 128, start=1.0, duration=duration)},
        )
    }
    table = FeatureTableCompiler(window_seconds=ROW).run(record, prior=prior).artifacts["feature_table"]
    row_start = table["time_seconds"].to_numpy()
    values = table["spec.frame_time"].to_numpy()
    assert np.all((values >= row_start) & (values < row_start + ROW))


# --- 2. hgb model family ---------------------------------------------------------------------------


def _toy_classification(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series(np.where(X["a"] > 1.0, 2, np.where(X["b"] > 0.5, 1, 0)))
    X.iloc[::17, 2] = np.nan  # HGB must accept missing values natively
    return X, y


def test_hgb_family_is_default_hgb_with_balanced_sample_weights():
    X, y = _toy_classification()
    model = build_model(model_family="hgb", task="classification", random_state=7)
    model.fit(X, y)
    reference = HistGradientBoostingClassifier(random_state=7)
    reference.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    assert isinstance(model.named_steps["model"], HistGradientBoostingClassifier)
    np.testing.assert_allclose(model.predict_proba(X), reference.predict_proba(X))


# --- 3. out-of-fold stage-1 probabilities ----------------------------------------------------------


class _RecordingEstimator:
    fitted_on: list[set] = []

    def __init__(self):
        self.subjects: set = set()

    def fit(self, X, y):
        self.subjects = set(X["subject_tag"].astype(int))
        self.classes_ = np.array(sorted(np.unique(y)))
        _RecordingEstimator.fitted_on.append(self.subjects)
        return self

    def predict_proba(self, X):
        leaked = X["subject_tag"].astype(int).isin(self.subjects).to_numpy()
        proba = np.full((len(X), len(self.classes_)), 1.0 / len(self.classes_))
        proba[leaked] = 0.0
        proba[leaked, 0] = 1.0  # marks rows predicted by a model that saw their subject
        return proba


def test_oof_probabilities_never_come_from_a_model_that_saw_the_subject(monkeypatch):
    script = _load_stage2_script()
    monkeypatch.setattr(tabular_runner, "build_model", lambda **kwargs: _RecordingEstimator())
    rng = np.random.default_rng(1)
    train = pd.DataFrame(
        {
            "subject_id": np.repeat([f"s{i}" for i in range(6)], 40),
            "subject_tag": np.repeat(np.arange(6), 40),
            "label": rng.integers(0, 3, size=240),
        }
    )
    oof = script.out_of_fold_probabilities(
        train,
        usable=["subject_tag"],
        target_column="label",
        subject_column="subject_id",
        class_labels=[0, 1, 2],
        model_family="hgb",
        random_state=7,
        n_folds=3,
    )
    np.testing.assert_allclose(oof, 1.0 / 3.0)
    assert len(_RecordingEstimator.fitted_on) == 3


def test_stage2_train_probs_auto_resolution():
    script = _load_stage2_script()
    assert script.resolve_stage2_train_probs("auto", "hgb") == "oof"
    assert script.resolve_stage2_train_probs("auto", "ensemble") == "insample"
    assert script.resolve_stage2_train_probs("oof", "ensemble") == "oof"


# --- 4. TCN balances classes once ------------------------------------------------------------------


@pytest.mark.parametrize("weighted_sampling", [True, False])
def test_tcn_uses_one_class_balancing_mechanism(monkeypatch, weighted_sampling):
    torch = pytest.importorskip("torch")
    from physionet_sleep.experiments import sequence_cnn

    captured = {}
    real_loss = torch.nn.CrossEntropyLoss

    def recording_loss(*args, **kwargs):
        captured["weight"] = kwargs.get("weight")
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(torch.nn, "CrossEntropyLoss", recording_loss)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 20, 3)).astype(np.float32)
    y = np.array([0] * 40 + [1] * 15 + [2] * 5)
    sequence_cnn._train_cnn_classifier(
        X, y, X_early_val=X[:12], y_early_val=y[:12], epochs=1, min_epochs=1, patience=1, batch_size=16,
        learning_rate=1e-3, hidden_channels=4, kernel_size=3, dropout=0.0, model_arch="dilated_tcn",
        history_steps=10, current_steps=5, future_steps=5, tcn_levels=2, weighted_sampling=weighted_sampling,
        random_state=0,
    )
    assert (captured["weight"] is None) == weighted_sampling
