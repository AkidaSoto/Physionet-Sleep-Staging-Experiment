
"""A compact temporal network over consecutively recorded feature epochs."""

from __future__ import annotations

import torch
from torch import nn


class ContextBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.temporal = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, groups=width)
        self.mix = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                 nn.Linear(width * 2, width), nn.Dropout(0.1))

    def forward(self, x: torch.Tensor, recorded: torch.Tensor) -> torch.Tensor:
        # Normalize and mask before convolution so padding never enters a recording.
        z = self.norm(x) * recorded
        z = self.temporal(z.transpose(1, 2)).transpose(1, 2)
        return (x + self.mix(z)) * recorded


class FeatureContextNet(nn.Module):
    def __init__(self, dilation: tuple[int, ...], width: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(90, width), nn.LayerNorm(width), nn.GELU())
        self.blocks = nn.ModuleList(ContextBlock(width, d) for d in dilation)
        self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 5))

    def forward(self, windows: torch.Tensor, recorded: torch.Tensor) -> torch.Tensor:
        mask = recorded.unsqueeze(-1)
        hidden = self.stem(windows) * mask
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.classifier(hidden)

