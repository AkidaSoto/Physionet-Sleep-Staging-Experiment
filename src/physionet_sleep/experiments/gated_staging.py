"""Directed temporal residuals over compact sleep-staging features."""

from __future__ import annotations

from math import log
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class StagingTerms(NamedTuple):
    logits: torch.Tensor
    local: torch.Tensor
    past: torch.Tensor
    future: torch.Tensor
    past_gate: torch.Tensor
    future_gate: torch.Tensor


class GatedStagingNet(nn.Module):
    """Score the target epoch and add separately gated past and future residuals."""

    def __init__(self, gated: bool, width: int = 64, radius: int = 8) -> None:
        super().__init__()
        if radius < 1:
            raise ValueError(
                "Context radius must include at least one neighboring epoch"
            )
        self.gated = gated
        self.radius = radius
        self.stem = nn.Sequential(nn.Linear(90, width), nn.LayerNorm(width), nn.GELU())
        self.local_classifier = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width, 5),
        )
        # Reverse the future branch in time so the same left padding works for
        # both directions. Separate channel groups learn distinct lag kernels.
        self.context = nn.Conv1d(
            2 * width, 2 * width, radius, groups=2 * width, bias=False
        )
        self.past_classifier = self._context_head(width)
        self.future_classifier = self._context_head(width)
        self.gates = nn.Linear(width, 10)
        nn.init.zeros_(self.gates.weight)
        nn.init.constant_(self.gates.bias, log(4))
        for head in (self.past_classifier, self.future_classifier):
            nn.init.zeros_(head[-1].weight)

    @staticmethod
    def _context_head(width: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width, 5, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        recorded: torch.Tensor,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | StagingTerms:
        valid = recorded.unsqueeze(-1)
        # Clear padded inputs before the stem to keep masked NaNs out of the
        # projection and its gradients.
        hidden = self.stem(x.masked_fill(~valid, 0)).masked_fill(~valid, 0)
        local = self.local_classifier(hidden).masked_fill(~valid, 0)
        channels = hidden.transpose(1, 2)
        directions = torch.cat((channels, channels.flip(-1)), dim=1)
        # Drop the current epoch from both directions. The extra left pad makes
        # the first output depend on a fully empty context.
        context = self.context(F.pad(directions[..., :-1], (self.radius, 0)))
        past_hidden, future_hidden = context.transpose(1, 2).chunk(2, dim=-1)
        future_hidden = future_hidden.flip(1)

        available = recorded.unsqueeze(1)
        support = torch.cat((available, available.flip(-1)), dim=1)
        support = F.max_pool1d(
            F.pad(support[..., :-1].to(hidden.dtype), (self.radius, 0)),
            self.radius,
            stride=1,
        ).bool()
        past_available = valid & support[:, 0, :].unsqueeze(-1)
        future_available = valid & support[:, 1, :].flip(-1).unsqueeze(-1)
        past = self.past_classifier(past_hidden).masked_fill(~past_available, 0)
        future = self.future_classifier(future_hidden).masked_fill(~future_available, 0)

        if self.gated:
            past_gate, future_gate = self.gates(hidden).sigmoid().chunk(2, dim=-1)
            past = past * past_gate
            future = future * future_gate
        elif return_components:
            past_gate, future_gate = torch.ones_like(past), torch.ones_like(future)
        full = local + past + future
        if return_components:
            return StagingTerms(full, local, past, future, past_gate, future_gate)
        return full


def stage_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weight: torch.Tensor,
    *,
    pair_strength: float = 0.0,
) -> torch.Tensor:
    """Weighted cross-entropy with an optional N1/N2 and N2/N3 pair margin."""
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    scored = targets >= 0
    logits = logits.masked_fill(~scored.unsqueeze(-1), 0)
    row_weight = class_weight[targets.clamp_min(0)] * scored
    loss = F.cross_entropy(logits, targets, ignore_index=-1, reduction="none")
    if pair_strength:
        n2_over_n1 = logits[:, 2] - logits[:, 1]
        n3_over_n2 = logits[:, 3] - logits[:, 2]
        margin = log(1.5)
        pair_loss = (
            (targets == 1) * F.softplus(n2_over_n1 + margin)
            + (targets == 2)
            * (F.softplus(-n2_over_n1 + margin) + F.softplus(n3_over_n2 + margin))
            / 2
            + (targets == 3) * F.softplus(-n3_over_n2 + margin)
        )
        loss = loss + pair_strength * pair_loss
    return (loss * row_weight).sum() / row_weight.sum().clamp_min(
        torch.finfo(row_weight.dtype).tiny
    )
