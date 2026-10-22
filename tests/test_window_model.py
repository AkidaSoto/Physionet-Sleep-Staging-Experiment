import numpy as np
import pandas as pd

from physionet_sleep.algorithms.eeg_tracks import FS, TRACKS, compute_tracks
from physionet_sleep.analysis.epoch_windows import WindowEncoder, robust_scale, subject_epochs
from physionet_sleep.decoders.hmm import bout_lengths, learn_transition_matrix, posterior_decode


def test_transition_matrix_counts_with_prior():
    tm = learn_transition_matrix([np.array([0, 0, 1, 1, 1])], n_states=2, prior=1.0)
    # from 0: one 0->0, one 0->1 (+1 prior each); from 1: two 1->1 (+1 prior), zero 1->0 (+1 prior)
    np.testing.assert_allclose(tm.trans, [[0.5, 0.5], [0.25, 0.75]])
    np.testing.assert_allclose(tm.initial, [2 / 3, 1 / 3])


def test_posterior_smooths_isolated_epoch_with_sticky_transitions():
    tm = learn_transition_matrix([np.r_[np.zeros(200, int), np.ones(200, int)]], n_states=2)
    proba = np.tile([0.8, 0.2], (21, 1))
    proba[10] = [0.4, 0.6]  # a single weak vote for state 1 inside a run of state 0
    post = posterior_decode(proba, tm)
    assert post.shape == proba.shape
    np.testing.assert_allclose(post.sum(1), 1.0)
    assert post[10].argmax() == 0


def test_bout_lengths():
    st, ln = bout_lengths(np.array([2, 2, 2, 0, 1, 1]))
    assert st.tolist() == [2, 0, 1] and ln.tolist() == [3, 1, 2]


def test_robust_scale_leaves_ratio_columns_raw():
    X = np.c_[np.arange(10.0), np.arange(10.0)]
    Z = robust_scale(X, ["spindle_features.delta_power", "epoch_ratio_features.rel_theta"])
    np.testing.assert_allclose(Z[:, 1], np.arange(10.0))
    assert abs(np.median(Z[:, 0])) < 1e-6


def test_window_steps_line_up_with_epoch_offsets():
    n = 300  # 10 epochs of 1-s values
    X = np.zeros((n, 1), np.float32)
    X[120:126] = 1.0  # first 6 s of epoch 4
    enc = WindowEncoder(["x"], context=2)
    enc.mean, enc.sd = np.zeros(1), np.ones(1)
    epochs = pd.DataFrame({"subject_id": ["s"] * 2, "epoch_index": [4, 5]})
    W = enc.transform({"s": X}, epochs)
    assert W.shape == (2, 25, 1)
    names = enc.column_names()
    assert names[0] == "x@-60s" and names[10] == "x@+0s" and names[-1] == "x@+84s"
    assert W[0, 10, 0] == 1.0 and W[0].sum() == 1.0  # epoch 4: at its own start
    assert W[1, 5, 0] == 1.0 and W[1].sum() == 1.0  # epoch 5: 30 s before


def test_window_pads_before_recording_start():
    enc = WindowEncoder(["x"], context=2)
    enc.mean, enc.sd = np.zeros(1), np.ones(1)
    W = enc.transform({"s": np.ones((90, 1), np.float32)}, pd.DataFrame({"subject_id": ["s"], "epoch_index": [0]}))
    assert W[0, :10].sum() == 0 and W[0, 10:25].sum() == 15  # two padded epochs, then three real ones


def test_subject_epochs_majority_label_and_alignment():
    rows = pd.DataFrame({"subject_id": "s", "time_seconds": np.arange(10) * 6.0, "slot": np.arange(10),
                         "stage_sec": [0, 0, 0, 2, 2, 3, 3, 3, -1, -1], "f": np.arange(10.0)})
    ep = subject_epochs(rows, ["f"])
    assert ep["epoch_index"].tolist() == [0, 1] and ep["label.stage_30s"].tolist() == [0, 3]
    np.testing.assert_allclose(ep["f"], [2.0, 7.0])


def test_tracks_pick_up_alpha_and_heart_rate():
    n = 60
    t = np.arange(n * FS) / FS
    alpha = 30 * np.sin(2 * np.pi * 10 * t)
    rng = np.random.default_rng(0)
    sig = {c: alpha + rng.normal(0, 1, t.size) for c in ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"]}
    sig.update({"E1-M2": rng.normal(0, 5, t.size), "Chin1-Chin2": rng.normal(0, 5, t.size),
                "ABD": np.sin(2 * np.pi * 0.25 * t), "CHEST": np.sin(2 * np.pi * 0.25 * t)})
    df = compute_tracks(sig, r_peak_seconds=np.arange(0, n, 1.0))
    assert list(df.columns) == ["second"] + TRACKS and len(df) == n
    assert df["c_rel_alpha"].iloc[2:-2].min() > 0.8
    np.testing.assert_allclose(df["hr"].iloc[1:-1], 60.0)


def test_transfer_tracks_match_cinc_tracks_on_shared_columns():
    from physionet_sleep.algorithms.eeg_tracks import TRANSFER_TRACKS, transfer_tracks

    n = 40
    t = np.arange(n * FS) / FS
    rng = np.random.default_rng(1)
    sig = {c: 20 * np.sin(2 * np.pi * 6 * t) + rng.normal(0, 2, t.size) for c in ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"]}
    sig.update({"E1-M2": rng.normal(0, 5, t.size), "Chin1-Chin2": rng.normal(0, 5, t.size),
                "ABD": np.sin(2 * np.pi * 0.25 * t), "CHEST": np.sin(2 * np.pi * 0.25 * t)})
    peaks = np.arange(0, n, 0.8)
    ref = compute_tracks(sig, peaks)
    new = transfer_tracks([(sig["C3-M2"], FS), (sig["C4-M1"], FS)], (sig["E1-M2"], FS), (sig["ABD"] + sig["CHEST"], FS), peaks)
    for col in [c for c in TRANSFER_TRACKS if not c.startswith("resp_")]:
        np.testing.assert_allclose(new[col].to_numpy(), ref[col].to_numpy(), rtol=1e-5, atol=1e-6)
