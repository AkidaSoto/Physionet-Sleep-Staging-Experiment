from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from physionet_sleep.experiments.waveform_data import EPOCH_SAMPLES, WaveformStore


def _write_night(root: Path, subject: str) -> None:
    folder = root / subject
    folder.mkdir()
    samples = np.zeros((2, EPOCH_SAMPLES + 2), np.int16)
    samples[:, 0] = [30, 150]
    samples[:, EPOCH_SAMPLES] = [-32768, 250]
    savemat(folder / f"{subject}.mat", {"val": samples}, format="4")
    (folder / f"{subject}.hea").write_text(
        f"{subject} 2 200 {samples.shape[1]}\n"
        f"{subject}.mat 16+24 2(10)/uV 16 0 30 0 0 Chin1-Chin2\n"
        f"{subject}.mat 16+24 1/uV 16 0 150 0 0 F3-M2\n",
        encoding="ascii",
    )


def test_waveform_alignment_calibration_and_masks(tmp_path):
    pytest.importorskip("wfdb")
    _write_night(tmp_path, "tr01-test")
    store = WaveformStore(tmp_path, permitted_subjects=["tr01-test"])
    window = store.window("tr01-test", 0)
    x, valid, channels = (window[name] for name in ("x", "valid", "channels"))

    assert x.shape == (5, 8, 6000)
    assert x.dtype == np.float32
    np.testing.assert_array_equal(valid.sum(axis=1), [0, 0, 6000, 2, 0])
    np.testing.assert_array_equal(np.flatnonzero(channels[2]), [0, 7])
    assert not channels[[0, 1, 4]].any()
    assert x[2, 0, 0] == pytest.approx(1.5)
    assert x[2, 7, 0] == pytest.approx(0.1)
    assert x[3, 0, 0] == pytest.approx(2.5)
    assert x[3, 7, 0] == 0
    with pytest.raises(ValueError, match="outside the permitted"):
        store.window("outside-manifest", 0)


def test_waveform_epoch_reuse_and_reconstruction_mask():
    torch = pytest.importorskip("torch")
    from physionet_sleep.experiments.waveform_resnet import WaveformResNet

    model = WaveformResNet().eval()
    rng = torch.Generator().manual_seed(7)
    # Two seconds per epoch is sufficient to test axis order and native patch masking.
    x = torch.randn(1, 5, 8, 400, generator=rng)
    valid = torch.ones(1, 5, 400, dtype=torch.bool)
    valid[:, 0] = False
    channels = torch.ones(1, 5, 8, dtype=torch.bool)
    channels[..., 1] = False
    patch_mask = torch.zeros(1, 5, 2, dtype=torch.bool)
    patch_mask[..., :1] = True
    changed = x.clone()
    changed[..., :200] += 100
    with torch.inference_mode():
        logits = model(x, valid, channels)
        embeddings = model.encode_epochs(x[0], valid[0], channels[0]).unsqueeze(0)
        torch.testing.assert_close(logits, model.classify_epochs(embeddings, valid.any(-1)))
        assert logits.shape == (1, 5)
        reconstructed = model.reconstruct(x, valid, channels, patch_mask)
        assert reconstructed.shape == x.shape
        torch.testing.assert_close(reconstructed, model.reconstruct(changed, valid, channels, patch_mask))
