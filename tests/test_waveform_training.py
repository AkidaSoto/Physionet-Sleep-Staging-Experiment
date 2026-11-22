import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from physionet_sleep.analysis.epoch_windows import SUBJ, TARGET
from physionet_sleep.experiments.waveform_training import fit_waveform_fold


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(8, 5)
        self.decoder = torch.nn.Linear(5, 8)
        self.dropout = torch.nn.Dropout(0.25)

    def encode_epochs(self, x, valid, channels):
        return self.encoder(x.mean(-1))

    def classify_epochs(self, latents, valid):
        return latents[:, 2]

    def forward(self, x, valid, channels):
        return self.dropout(self.classify_epochs(self.encode_epochs(x, valid, channels), valid))

    def reconstruct(self, x, valid, channels, patch_mask):
        visible = x.masked_fill(patch_mask.repeat_interleave(200, -1).unsqueeze(-2), 0)
        latents = self.dropout(self.encode_epochs(visible, valid, channels))
        return self.decoder(latents).unsqueeze(-1).expand_as(x)


class _Store:
    def __init__(self, pause=None, pause_on=None):
        self.pause = pause
        self.pause_on = pause_on
        self.calls = 0

    def epochs(self, subject, epochs):
        values = np.broadcast_to(np.asarray(epochs, dtype=np.float32)[:, None, None] / 10,
                                 (len(epochs), 8, 6000)).copy()
        return values, np.ones((len(epochs), 6000), bool), np.ones((len(epochs), 8), bool)

    def window(self, subject, epoch):
        self.calls += 1
        if self.calls == self.pause_on:
            self.pause.touch()
        return dict(zip(("x", "valid", "channels"), self.epochs(subject, range(epoch - 2, epoch + 3))))


def test_waveform_training_resumes_the_same_updates(tmp_path):
    train = pd.DataFrame({SUBJ: ["train"] * 5, "epoch_index": range(5), TARGET: range(5)})
    valid = train.assign(**{SUBJ: "valid"})
    test = train.assign(**{SUBJ: "test"}).iloc[::-1]
    settings = dict(pretrained=True, device="cpu", batch_size=1, seed=7, pretrain_steps=2, supervised_steps=2)

    pause_file = tmp_path / "pause"
    paused = fit_waveform_fold(_Store(pause_file, pause_on=1), _Model, train, valid, test,
                               output_dir=tmp_path / "resumed", pause_file=pause_file, **settings)
    assert paused is None
    pause_file.unlink()
    resumed = fit_waveform_fold(_Store(pause_file, pause_on=2), _Model, train, valid, test,
                                output_dir=tmp_path / "resumed", pause_file=pause_file, **settings)
    assert resumed is None
    pause_file.unlink()
    resumed = fit_waveform_fold(_Store(), _Model, train, valid, test,
                                output_dir=tmp_path / "resumed", **settings)
    uninterrupted = fit_waveform_fold(_Store(), _Model, train, valid, test,
                                      output_dir=tmp_path / "uninterrupted", **settings)
    np.testing.assert_array_equal(resumed, uninterrupted)
    assert resumed.shape == (5, 5)
    np.testing.assert_allclose(resumed.sum(1), 1.0, atol=1e-6)

    recovered = fit_waveform_fold(_Store(), _Model, train, valid, test,
                                  output_dir=tmp_path / "resumed", **settings)
    np.testing.assert_array_equal(resumed, recovered)
    with pytest.raises(ValueError, match="configuration changed"):
        fit_waveform_fold(_Store(), _Model, train.iloc[::-1], valid, test,
                          output_dir=tmp_path / "resumed", **settings)
