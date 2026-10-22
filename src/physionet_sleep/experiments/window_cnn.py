"""Attention CNN over 1-s windows (the neural comparison for the window staging model).

Input: (batch, seconds, signals) windows of the target epoch and its neighbours, normalised as in
analysis.epoch_windows. A 1x1 projection to 32 channels, three residual dilated convolution blocks (dilation 1, 2, 4;
kernel 5), and attention pooling: each second gets a softmax weight, and the weighted sum of its features is
classified. The attention weights and gradient x input give a per-second, per-signal explanation of each decision.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 5, dilation=dilation, padding=2 * dilation)
        self.norm = nn.LayerNorm(ch)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        h = self.drop(nn.functional.gelu(self.conv(x)))
        return self.norm((x + h).transpose(1, 2)).transpose(1, 2)


class EpochNet(nn.Module):
    def __init__(self, n_signals: int, ch: int = 32, n_classes: int = 5):
        super().__init__()
        self.inp = nn.Conv1d(n_signals, ch, 1)
        self.blocks = nn.Sequential(Block(ch, 1), Block(ch, 2), Block(ch, 4))
        self.score = nn.Conv1d(ch, 1, 1)
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x, return_attn: bool = False):  # x: (B, T, K)
        h = self.blocks(self.inp(x.transpose(1, 2)))
        a = torch.softmax(self.score(h).squeeze(1), dim=-1)  # (B, T)
        out = self.head((h * a[:, None, :]).sum(-1))
        return (out, a) if return_attn else out


def explain(models: list[EpochNet], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seed-averaged probabilities, attention (B, T) and gradient x input (B, T, K) for the predicted stage."""
    xt = torch.from_numpy(np.asarray(x, np.float32)).requires_grad_(True)
    with torch.no_grad():
        prob = np.mean([torch.softmax(m(xt), -1).numpy() for m in models], 0)
    cls = torch.from_numpy(prob.argmax(1))
    sal, att = 0.0, 0.0
    for m in models:
        out, a = m(xt, return_attn=True)
        (g,) = torch.autograd.grad(out.gather(1, cls[:, None]).sum(), xt)
        sal = sal + (g * xt).detach().numpy() / len(models)
        att = att + a.detach().numpy() / len(models)
    return prob, att, sal
