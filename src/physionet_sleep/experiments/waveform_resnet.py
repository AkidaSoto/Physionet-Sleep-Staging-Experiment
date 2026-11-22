"""A shared per-epoch 1D ResNet with a five-epoch staging head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from physionet_sleep.experiments.waveform_data import CHANNELS, SAMPLE_RATE

_STEM_STRIDE = 8
_ENCODER_STRIDE = 32


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=2, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 5, padding=2, bias=False),
            nn.GroupNorm(4, out_channels),
        )
        self.skip = (
            nn.Identity() if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.layers(x) + self.skip(x))


class WaveformResNet(nn.Module):
    """Stage a center epoch from 150 seconds of EEG, EOG, and chin EMG."""

    def __init__(self):
        super().__init__()
        n_channels = len(CHANNELS)
        self.stem = nn.Conv1d(n_channels, 16, 33, stride=_STEM_STRIDE, padding=16, bias=False)
        self.lead_embedding = nn.Linear(n_channels, 16, bias=False)
        self.mask_embedding = nn.Parameter(torch.zeros(16))
        self.encoder = nn.Sequential(
            _ResidualBlock(16, 16),
            _ResidualBlock(16, 24, stride=2),
            _ResidualBlock(24, 32, stride=2),
        )
        self.context = nn.Sequential(nn.Conv1d(33, 32, 3, padding=1), nn.GELU(), nn.Conv1d(32, 32, 3, padding=1))
        self.classifier = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 5))
        self.decoder = nn.Conv1d(32, n_channels * _ENCODER_STRIDE, 1)

    @property
    def staging_parameter_count(self) -> int:
        """Count scalar weights used when predicting sleep stages."""
        total = sum(parameter.numel() for parameter in self.parameters())
        reconstruction_only = self.mask_embedding.numel() + sum(parameter.numel() for parameter in self.decoder.parameters())
        return total - reconstruction_only

    def _encode(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        channels: torch.Tensor,
        masked: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visible = valid if masked is None else valid & ~masked
        x = x.masked_fill(~(visible[:, None, :] & channels[:, :, None]), 0)
        stem = self.stem(x)
        stem = stem + self.lead_embedding(channels.to(dtype=x.dtype))[:, :, None]
        if masked is not None:
            hidden_tokens = masked[:, None, ::_STEM_STRIDE].to(dtype=stem.dtype)
            stem = stem + hidden_tokens * self.mask_embedding[None, :, None]
        return self.encoder(stem)

    def encode_epochs(self, x: torch.Tensor, valid: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
        """Encode each epoch independently so contiguous inference can reuse embeddings."""
        encoded = self._encode(x, valid, channels)
        weights = valid[:, None, ::_ENCODER_STRIDE]
        return (encoded * weights).sum(dim=-1) / weights.sum(dim=-1).clamp(min=1)

    def classify_epochs(self, latents: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Read five adjacent epoch embeddings and predict the center stage."""
        latent = latents.masked_fill(~valid[:, :, None], 0).transpose(1, 2)
        context = self.context(torch.cat((latent, valid[:, None, :].to(dtype=latent.dtype)), dim=1))
        center = latent.shape[-1] // 2
        return self.classifier((latent + context)[:, :, center])

    def forward(self, x: torch.Tensor, valid: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
        n_batch, n_epochs, n_channels, n_samples = x.shape
        encoded = self.encode_epochs(
            x.reshape(-1, n_channels, n_samples),
            valid.reshape(-1, n_samples),
            channels.reshape(-1, n_channels),
        )
        return self.classify_epochs(encoded.reshape(n_batch, n_epochs, -1), valid.any(-1) & channels.any(-1))

    def reconstruct(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
        channels: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct native samples after hiding the indicated one-second patches."""
        n_batch, n_epochs, n_channels, n_samples = x.shape
        masked = patch_mask.reshape(n_batch * n_epochs, -1).repeat_interleave(SAMPLE_RATE, dim=-1)
        encoded = self._encode(
            x.reshape(-1, n_channels, n_samples),
            valid.reshape(-1, n_samples),
            channels.reshape(-1, n_channels),
            masked[:, :n_samples],
        )
        decoded = self.decoder(encoded).reshape(n_batch, n_epochs, n_channels, _ENCODER_STRIDE, -1)
        return decoded.transpose(-2, -1).flatten(-2)[..., :n_samples]
