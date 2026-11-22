import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from physionet_sleep.analysis.epoch_windows import SUBJ, TARGET
from physionet_sleep.experiments.waveform_subepoch import (
    ContiguousCohort,
    SubEpochWaveformNet,
    calibrate_prior_power,
    fit_subepoch_fold,
    oof_prior_power_calibration,
    predict_subepoch,
    tempered_class_weights,
)


class _FakeStore:
    def __init__(self, n_samples: int = 6000):
        self.n_samples = n_samples

    def epochs(self, subject: str, epoch_indices):
        epochs = list(epoch_indices)
        rng = np.random.default_rng(abs(hash(subject)) % (2**31))
        base = rng.normal(0.0, 0.5, size=(1, 8, self.n_samples)).astype(np.float32)
        offsets = np.asarray(epochs, dtype=np.float32)[:, None, None] * 0.05
        x = np.broadcast_to(base, (len(epochs), 8, self.n_samples)).copy() + offsets
        valid = np.ones((len(epochs), self.n_samples), dtype=bool)
        channels = np.ones((len(epochs), 8), dtype=bool)
        channels[:, 2] = False
        return x, valid, channels


def test_subepoch_net_shapes_and_mask_invariance():
    torch.manual_seed(7)
    model = SubEpochWaveformNet().eval()
    x = torch.randn(2, 4, 8, 6000)
    valid = torch.ones(2, 4, 6000, dtype=torch.bool)
    valid[:, 0, :1200] = False
    valid[:, 3] = False
    channels = torch.ones(2, 4, 8, dtype=torch.bool)
    channels[:, :, 1] = False

    changed = x.clone()
    changed[:, 0, :, :1200] += 50.0
    changed[:, 3] -= 25.0
    changed[:, :, 1, :] += 100.0

    with torch.inference_mode():
        tokens, tok_valid = model.encode_subepochs(x, valid, channels)
        assert tokens.shape == (2, 4, 5, 64)
        assert tok_valid.shape == (2, 4, 5)
        assert not tok_valid[:, 0, 0].any()
        assert not tok_valid[:, 3].any()

        logits, local = model(x, valid, channels, return_local=True)
        assert logits.shape == (2, 4, 5)
        assert local.shape == (2, 4, 5)

        logits_changed = model(changed, valid, channels)
        torch.testing.assert_close(logits, logits_changed, rtol=0.0, atol=0.0)

    with pytest.raises(ValueError, match="divisible by 600"):
        model.encode_subepochs(torch.zeros(1, 2, 8, 500), torch.ones(1, 2, 500, dtype=bool), torch.ones(1, 2, 8, dtype=bool))


def test_predict_subepoch_matches_full_forward_with_gaps_and_row_order():
    torch.manual_seed(11)
    model = SubEpochWaveformNet().eval()
    store = _FakeStore()
    # Non-contiguous epochs starting at 10, returned in scrambled row order.
    frame = pd.DataFrame({
        SUBJ: ["night_a"] * 4,
        "epoch_index": [14, 10, 12, 11],
        TARGET: [0, 1, 2, 3],
    })
    cohort = ContiguousCohort.from_frame(frame)
    chunked_probs = predict_subepoch(store, model, cohort, batch_epochs=2, device="cpu")

    x_full, v_full, c_full = store.epochs("night_a", range(10, 15))
    with torch.inference_mode():
        logits = model(
            torch.from_numpy(x_full).unsqueeze(0),
            torch.from_numpy(v_full).unsqueeze(0),
            torch.from_numpy(c_full).unsqueeze(0),
        ).squeeze(0)
        direct_probs = logits.softmax(-1).numpy()

    expected = direct_probs[np.array([4, 0, 2, 1])]
    np.testing.assert_allclose(chunked_probs, expected, rtol=1e-5, atol=1e-6)


def test_fit_subepoch_fold_resume_and_guards(tmp_path):
    train = pd.DataFrame({
        SUBJ: ["tr_a"] * 6 + ["tr_b"] * 5,
        "epoch_index": [10, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9],
        TARGET: [0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4],
    })
    valid = pd.DataFrame({
        SUBJ: ["val_a"] * 5,
        "epoch_index": [2, 3, 4, 5, 6],
        TARGET: [0, 1, 2, 3, 4],
    })
    test = pd.DataFrame({
        SUBJ: ["te_a"] * 4,
        "epoch_index": [8, 5, 7, 6],
        TARGET: [4, 0, 2, 1],
    })
    store = _FakeStore()
    opts = dict(
        device="cpu",
        batch_size=2,
        chunk_epochs=4,
        eval_batch_epochs=2,
        max_steps=4,
        validation_interval=2,
        checkpoint_interval_steps=100,
        seed=7,
    )

    paused = fit_subepoch_fold(
        store, train, valid, test,
        output_dir=tmp_path / "resumed", smoke_pause_after=2, **opts,
    )
    assert paused is None

    resumed = fit_subepoch_fold(
        store, train, valid, test,
        output_dir=tmp_path / "resumed", **opts,
    )
    uninterrupted = fit_subepoch_fold(
        store, train, valid, test,
        output_dir=tmp_path / "uninterrupted", **opts,
    )
    assert resumed is not None and uninterrupted is not None
    res_test, res_val, res_rt = resumed
    un_test, un_val, _ = uninterrupted

    np.testing.assert_allclose(res_test, un_test, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(res_val, un_val, rtol=0.0, atol=0.0)
    assert res_test.shape == (4, 5)
    assert np.isfinite(res_test).all()
    np.testing.assert_allclose(res_test.sum(axis=1), 1.0, atol=1e-6)
    assert res_rt["best_step"] in {2, 4}

    ckpt_res = torch.load(tmp_path / "resumed/checkpoint.pt", map_location="cpu", weights_only=True)
    ckpt_un = torch.load(tmp_path / "uninterrupted/checkpoint.pt", map_location="cpu", weights_only=True)
    for k, v in ckpt_res["best_model"].items():
        torch.testing.assert_close(v, ckpt_un["best_model"][k], rtol=0.0, atol=0.0)

    # Complete checkpoint reload returns identical predictions without retraining.
    recovered = fit_subepoch_fold(store, train, valid, test, output_dir=tmp_path / "resumed", **opts)
    assert recovered is not None
    np.testing.assert_array_equal(recovered[0], res_test)

    with pytest.raises(ValueError, match="disjoint"):
        fit_subepoch_fold(store, train, train.copy(), test, output_dir=tmp_path / "overlap", **opts)
    with pytest.raises(ValueError, match="configuration changed"):
        fit_subepoch_fold(store, train.iloc[::-1], valid, test, output_dir=tmp_path / "resumed", **opts)


def test_tempered_weights_and_oof_prior_calibration():
    labels = np.array([0, 0, 0, 0, 1, 2, 2, 3, 4], dtype=np.int64)
    counts, w = tempered_class_weights(labels, power=-0.50)
    np.testing.assert_array_equal(counts, [4, 1, 2, 1, 1])
    assert w[0].item() == pytest.approx(0.5 * w[1].item(), rel=1e-5)
    assert w.mean().item() == pytest.approx(1.0, rel=1e-5)

    rng = np.random.default_rng(7)
    probs = rng.dirichlet([1.0] * 5, size=50).astype(np.float32)
    priors = np.tile(np.array([0.3, 0.1, 0.35, 0.1, 0.15]), (50, 1))
    np.testing.assert_allclose(calibrate_prior_power(probs, priors, 0.0), probs, atol=1e-6)

    folds = np.repeat(np.arange(5), 10)
    y = rng.integers(0, 5, size=50)
    fold_counts = {f: np.array([300, 100, 350, 100, 150]) for f in range(5)}
    cal = oof_prior_power_calibration(y, folds, probs, fold_counts)
    for key in ("bayes_half_probabilities", "nested_probabilities"):
        arr = cal[key]
        assert arr.shape == (50, 5)
        assert np.isfinite(arr).all() and (arr >= 0).all()
        np.testing.assert_allclose(arr.sum(axis=1), 1.0, atol=1e-5)
