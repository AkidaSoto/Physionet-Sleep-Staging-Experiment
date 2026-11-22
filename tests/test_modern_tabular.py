import numpy as np
import pandas as pd
import pytest

from physionet_sleep.analysis.epoch_windows import SPINDLE_BURSTS, SUBJ, TARGET, WindowEncoder, dense_signals
from physionet_sleep.experiments.modern_tabular import REMOVED_TRACKS, fit_feature_reduction_fold
from physionet_sleep.experiments.tabular_runner import build_model


TRACKS = ["c_rel_alpha", "c_rel_beta", "c_rel_theta", "c_log_power", "o_rel_alpha", "eog_high", *REMOVED_TRACKS]
FEATURES = [*(f"feature_{i}" for i in range(11)), SPINDLE_BURSTS]


def test_reduction_matches_window_encoding_and_preserves_test_order(tmp_path):
    rng = np.random.default_rng(7)
    dense = {}
    frames = {}
    (tmp_path / "tracks1s").mkdir()
    (tmp_path / "rows6").mkdir()
    reduced_tracks = [name for name in TRACKS if name not in REMOVED_TRACKS]
    stages = np.tile(np.repeat([0, 4], 10), 2)
    n_seconds = len(stages) * 30
    for subject in ("b", "a", "test-2", "test-1"):
        tracks = pd.DataFrame(rng.uniform(0.1, 1.0, (n_seconds, len(TRACKS))), columns=TRACKS)
        tracks["c_log_power"] += np.repeat(stages, 30)
        rows = pd.DataFrame(rng.normal(size=(n_seconds // 6, 11)), columns=FEATURES[:-1])
        rows["slot"] = np.arange(len(rows))
        tracks.to_parquet(tmp_path / "tracks1s" / f"{subject}.parquet", index=False)
        rows.to_parquet(tmp_path / "rows6" / f"{subject}.parquet", index=False)
        dense[subject] = dense_signals(tracks, rows, reduced_tracks, FEATURES)
        frames[subject] = pd.DataFrame({SUBJ: subject, "epoch_index": np.arange(len(stages)), TARGET: stages})
    train = pd.concat([frames["b"], frames["a"]], ignore_index=True)
    test = pd.concat([frames["test-2"], frames["test-1"]], ignore_index=True).sample(frac=1.0, random_state=7)

    probabilities, metadata = fit_feature_reduction_fold(tmp_path, train, test, TRACKS, FEATURES)

    encoder = WindowEncoder(reduced_tracks + FEATURES).fit(dense, ["a", "b"])
    columns = encoder.column_names()
    reference = build_model(model_family="hgb", task="classification", random_state=7)
    reference.fit(pd.DataFrame(encoder.transform(dense, train).reshape(len(train), 450), columns=columns), train[TARGET])
    expected = reference.predict_proba(
        pd.DataFrame(encoder.transform(dense, test).reshape(len(test), 450), columns=columns)
    )
    np.testing.assert_allclose(probabilities[:, [0, 4]], expected)
    np.testing.assert_array_equal(probabilities[:, 1:4], 0)
    assert metadata["input_signals"] == 18 and metadata["input_columns"] == 450
    assert not metadata["raw_extraction_changed"]


def test_reduction_rejects_overlapping_patients_before_loading_caches(tmp_path):
    epochs = pd.DataFrame({SUBJ: ["same-patient"], "epoch_index": [0], TARGET: [2]})
    with pytest.raises(ValueError, match="patients overlap"):
        fit_feature_reduction_fold(tmp_path, epochs, epochs, TRACKS, FEATURES)
