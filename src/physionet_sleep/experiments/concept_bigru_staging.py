"""Concept-additive whole-night BiGRU sleep staging and conformal uncertainty (#79)."""

from __future__ import annotations

from math import ceil, log
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from physionet_sleep.experiments.bigru_staging import (
    NIGHT_HISTORY_COLUMNS,
    SLOW_WAVE_PANEL_COLUMNS,
    calibrate_prior_power,
    fit_prior_power_oof,
)

STEP_PER_EPOCH = 5
CONCEPT_NAMES: tuple[str, ...] = ("slow_wave", "n2_events", "wake_n1", "rem_tone")

# Non-redundant 6-s step signals (13 signals -> 65 step inputs, zero epoch duplicates)
# grouped strictly by clinical/physiological concept system.
CONCEPT_STEP_SIGNALS: dict[str, tuple[str, ...]] = {
    "slow_wave": (
        "spindle_features.delta_power",
        "spindle_features.aperiodic_exponent",
    ),
    "n2_events": (
        "c_rel_sigma",
        "sigma_ratio",
    ),
    "wake_n1": (
        "o_rel_alpha",
        "c_rel_alpha",
        "c_rel_theta",
        "o_rel_theta",
        "c_rel_beta",
        "c_log_power",
    ),
    "rem_tone": (
        "eog_high",
        "eye_movement_activity.eye_movement_activity",
        "eye_movement_activity.eye_movement_baseline",
    ),
}

# Unique 30-s epoch features (16 inputs: 4 non-duplicated rows6 + 12 #60/#67/#51 summaries).
# Drops 5x-repeated epoch values, o_theta_30s, n2_recent_event_support, elapsed_squared,
# and redundant spectral ratios already represented cleanly at 6s.
CONCEPT_EPOCH_SIGNALS: dict[str, tuple[str, ...]] = {
    "slow_wave": (
        *SLOW_WAVE_PANEL_COLUMNS,
        "slow_burden_vs_clock",
        "slow_coverage_prior_30min",
    ),
    "n2_events": (
        "spindle_bursts",
        "kc_shape_events",
        "recent_k_support",
        "recent_spindle_support",
    ),
    "wake_n1": (
        "elapsed_hours",
        "fraction_elapsed",
    ),
    "rem_tone": (
        "epoch_ratio_features.eog_rem_vs_noise_db",
        "epoch_ratio_features.eog_rem_vs_sem_db",
        "epoch_ratio_features.emg_floor_db",
        "atonia_fraction_10min",
        "nearby_eye_atonia",
    ),
}

# Validate against upstream constants so no column drifts silently.
assert set(SLOW_WAVE_PANEL_COLUMNS).issubset(CONCEPT_EPOCH_SIGNALS["slow_wave"])
assert {"elapsed_hours", "fraction_elapsed", "slow_burden_vs_clock", "slow_coverage_prior_30min", "atonia_fraction_10min", "nearby_eye_atonia"}.issubset(
    set(NIGHT_HISTORY_COLUMNS)
)


def build_concept_layout(
    step_per_epoch: int = STEP_PER_EPOCH,
    step_signals: dict[str, tuple[str, ...]] | None = None,
    epoch_signals: dict[str, tuple[str, ...]] | None = None,
    concept_names: tuple[str, ...] = CONCEPT_NAMES,
) -> tuple[tuple[str, ...], dict[str, slice], tuple[int, ...]]:
    """Build deterministic concept-contiguous feature names, slices, and widths."""
    step_signals = step_signals or CONCEPT_STEP_SIGNALS
    epoch_signals = epoch_signals or CONCEPT_EPOCH_SIGNALS
    names: list[str] = []
    slices: dict[str, slice] = {}
    dims: list[int] = []
    offset = 0
    step_sec = 30 // step_per_epoch
    for concept in concept_names:
        start = offset
        for sig in step_signals.get(concept, ()):
            for step_idx in range(step_per_epoch):
                names.append(f"{sig}@+{step_idx * step_sec}s")
        for sig in epoch_signals.get(concept, ()):
            names.append(sig)
        width = len(names) - start
        slices[concept] = slice(start, start + width)
        dims.append(width)
        offset += width
    return tuple(names), slices, tuple(dims)


FEATURE_NAMES_79, CONCEPT_SLICES, DEFAULT_CONCEPT_DIMS = build_concept_layout()
TOTAL_CONCEPT_DIM = len(FEATURE_NAMES_79)  # 13*5 + 16 = 81 non-redundant inputs


class ConceptTerms(NamedTuple):
    logits: torch.Tensor
    local: torch.Tensor
    past: torch.Tensor
    future: torch.Tensor
    past_gate: torch.Tensor
    future_gate: torch.Tensor
    concept_logits: torch.Tensor
    local_bias: torch.Tensor


class ConceptBiGRUStagingNet(nn.Module):
    """Concept-additive physiological stem + 1-layer causal/anti-causal gated BiGRU.

    Each physiological concept group has its own private stem encoder and zero-centered
    linear logit head (no cross-concept interaction before temporal context). The four
    additive concept logits plus an explicit stage bias form `local`; the concatenated
    concept embeddings feed unidirectional past and future GRUs modulated by bounded
    per-stage sigmoid gates.
    """

    def __init__(
        self,
        concept_dims: tuple[int, ...] = DEFAULT_CONCEPT_DIMS,
        concept_names: tuple[str, ...] = CONCEPT_NAMES,
        width_per_concept: int | tuple[int, ...] = 16,
        dropout: float = 0.15,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if len(concept_dims) != len(concept_names):
            raise ValueError("concept_dims and concept_names must align")
        if any(d <= 0 for d in concept_dims):
            raise ValueError(f"Every concept must have positive input width: {concept_dims}")
        self.concept_dims = tuple(int(d) for d in concept_dims)
        self.concept_names = tuple(concept_names)
        self.input_dim = sum(self.concept_dims)
        if isinstance(width_per_concept, int):
            self.concept_widths = tuple(int(width_per_concept) for _ in self.concept_dims)
            self.width_per_concept = int(width_per_concept)
        else:
            self.concept_widths = tuple(int(w) for w in width_per_concept)
            if len(self.concept_widths) != len(self.concept_dims):
                raise ValueError("width_per_concept tuple must align with concept_dims")
            self.width_per_concept = self.concept_widths[0]
        self.width = sum(self.concept_widths)
        self.dropout = float(dropout)
        self.num_layers = int(num_layers)

        self.concept_stems = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, c_width),
                    nn.LayerNorm(c_width),
                    nn.GELU(),
                )
                for dim, c_width in zip(self.concept_dims, self.concept_widths)
            ]
        )
        self.concept_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(c_width, 5, bias=False),
                )
                for c_width in self.concept_widths
            ]
        )
        self.local_bias = nn.Parameter(torch.zeros(5))

        rnn_dropout = dropout if self.num_layers > 1 else 0.0
        self.past_rnn = nn.GRU(
            self.width,
            self.width,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.future_rnn = nn.GRU(
            self.width,
            self.width,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.past_classifier = self._context_head(self.width, dropout)
        self.future_classifier = self._context_head(self.width, dropout)
        for head in (self.past_classifier, self.future_classifier):
            nn.init.zeros_(head[-1].weight)

        self.gates = nn.Linear(self.width, 10)
        nn.init.zeros_(self.gates.weight)
        nn.init.constant_(self.gates.bias, log(4))

    @staticmethod
    def _context_head(width: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 5, bias=False),
        )

    def _run_anti_causal(
        self,
        hidden: torch.Tensor,
        recorded: torch.Tensor,
        valid_len: torch.Tensor,
        lengths_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Flip each night's valid prefix so batch padding never leaks into future RNN init."""
        seq_len = hidden.shape[1]
        rev_rows = []
        for b in range(hidden.shape[0]):
            length = int(valid_len[b].item())
            if length > 0:
                flipped = hidden[b, :length].flip(0)
                if length < seq_len:
                    flipped = F.pad(flipped, (0, 0, 0, seq_len - length))
            else:
                flipped = hidden.new_zeros((seq_len, hidden.shape[-1]))
            rev_rows.append(flipped)
        rev_hidden = torch.stack(rev_rows, dim=0)
        future_in = F.pad(rev_hidden[:, :-1], (0, 0, 1, 0))
        packed_future = nn.utils.rnn.pack_padded_sequence(
            future_in, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_future_out, _ = self.future_rnn(packed_future)
        future_rev, _ = nn.utils.rnn.pad_packed_sequence(
            packed_future_out, batch_first=True, total_length=seq_len
        )
        unrev_rows = []
        for b in range(hidden.shape[0]):
            length = int(valid_len[b].item())
            if length > 0:
                restored = future_rev[b, :length].flip(0)
                if length < seq_len:
                    restored = F.pad(restored, (0, 0, 0, seq_len - length))
            else:
                restored = future_rev.new_zeros((seq_len, future_rev.shape[-1]))
            unrev_rows.append(restored)
        future_hidden = torch.stack(unrev_rows, dim=0)
        future_count = (
            F.pad(recorded[:, 1:].long(), (0, 1)).flip(1).cumsum(dim=1).flip(1)
        )
        future_available = recorded.unsqueeze(-1) & (future_count > 0).unsqueeze(-1)
        return self.future_classifier(future_hidden).masked_fill(~future_available, 0)

    def forward(
        self,
        x: torch.Tensor,
        recorded: torch.Tensor,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | ConceptTerms:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected feature width {self.input_dim}, received {x.shape[-1]}")
        valid = recorded.unsqueeze(-1)
        x_clean = x.masked_fill(~valid, 0)

        h_parts: list[torch.Tensor] = []
        concept_logits_list: list[torch.Tensor] = []
        offset = 0
        for dim, stem, head in zip(
            self.concept_dims, self.concept_stems, self.concept_heads
        ):
            x_k = x_clean[..., offset : offset + dim]
            h_k = stem(x_k).masked_fill(~valid, 0)
            c_k = head(h_k).masked_fill(~valid, 0)
            h_parts.append(h_k)
            concept_logits_list.append(c_k)
            offset += dim

        concept_logits = torch.stack(concept_logits_list, dim=-2)  # (B, T, K, 5)
        local_bias = self.local_bias.view(1, 1, 5)
        local = (concept_logits.sum(dim=-2) + local_bias).masked_fill(~valid, 0)
        hidden = torch.cat(h_parts, dim=-1).masked_fill(~valid, 0)

        seq_len = hidden.shape[1]
        if seq_len <= 1:
            past_raw = torch.zeros_like(local)
            future_raw = torch.zeros_like(local)
        else:
            indices = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
            valid_len = (recorded.long() * (indices + 1)).max(dim=1).values
            lengths_cpu = valid_len.clamp_min(1).cpu()

            past_in = F.pad(hidden[:, :-1], (0, 0, 1, 0))
            packed_past = nn.utils.rnn.pack_padded_sequence(
                past_in, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_past_out, _ = self.past_rnn(packed_past)
            past_hidden, _ = nn.utils.rnn.pad_packed_sequence(
                packed_past_out, batch_first=True, total_length=seq_len
            )
            prior_count = F.pad(recorded[:, :-1].long(), (1, 0)).cumsum(dim=1)
            past_available = valid & (prior_count > 0).unsqueeze(-1)
            past_raw = self.past_classifier(past_hidden).masked_fill(~past_available, 0)

            future_raw = self._run_anti_causal(hidden, recorded, valid_len, lengths_cpu)

        past_gate, future_gate = self.gates(hidden).sigmoid().chunk(2, dim=-1)
        past = past_raw * past_gate
        future = future_raw * future_gate
        logits = (local + past + future).masked_fill(~valid, 0)
        if return_components:
            return ConceptTerms(
                logits=logits,
                local=local,
                past=past,
                future=future,
                past_gate=past_gate,
                future_gate=future_gate,
                concept_logits=concept_logits,
                local_bias=self.local_bias.detach().clone(),
            )
        return logits


def smoothed_stage_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weight: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Weighted cross-entropy over scored epochs with optional label smoothing."""
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
    mask = targets >= 0
    if not mask.any():
        return logits.sum() * 0.0
    scored_logits = logits[mask]
    scored_targets = targets[mask]
    sample_w = class_weight[scored_targets]
    log_probs = F.log_softmax(scored_logits, dim=-1)
    nll = -log_probs.gather(-1, scored_targets.unsqueeze(-1)).squeeze(-1)
    if label_smoothing > 0.0:
        smooth = -log_probs.mean(dim=-1)
        per_epoch = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    else:
        per_epoch = nll
    return (per_epoch * sample_w).sum() / sample_w.sum().clamp_min(1e-12)


class EMAWeights:
    """Exponential moving average shadow weights with early-step bias warmup."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.updates = 0
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        self.updates += 1
        eff_decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        one_minus = 1.0 - eff_decay
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(eff_decay).add_(param.detach(), alpha=one_minus)

    def copy_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Swap EMA weights onto `model` and return previous state_dict tensors to restore."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            state = model.state_dict()
            for name, val in self.shadow.items():
                state[name].copy_(val)
        return backup

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {k: v.detach().clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])
        self.shadow = {k: v.detach().clone() for k, v in state["shadow"].items()}


def fit_split_conformal_threshold(
    probs: np.ndarray, y: np.ndarray, alpha: float
) -> float:
    """Least Ambiguous Classifier (LAC) split conformal quantile on score 1 - p_y."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(y)
    if n == 0:
        raise ValueError("Cannot calibrate conformal threshold on empty array")
    scores = 1.0 - probs[np.arange(n), y]
    q_level = min(1.0, ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def fit_mondrian_conformal_thresholds(
    probs: np.ndarray, y: np.ndarray, alpha: float, n_classes: int = 5
) -> np.ndarray:
    """Class-conditional (Mondrian) LAC conformal thresholds per stage."""
    thresholds = np.empty(n_classes, dtype=np.float64)
    for cls in range(n_classes):
        mask = y == cls
        if not mask.any():
            raise ValueError(f"Missing class {cls} in Mondrian conformal calibration set")
        thresholds[cls] = fit_split_conformal_threshold(probs[mask], y[mask], alpha)
    return thresholds


def conformal_prediction_sets(
    probs: np.ndarray,
    thresholds: float | np.ndarray,
    *,
    guarantee_nonempty: bool = True,
) -> tuple[np.ndarray, float]:
    """Return boolean prediction-set matrix `(N, C)` and pre-fallback empty fraction."""
    nonconf = 1.0 - probs
    thresh_arr = np.asarray(thresholds, dtype=np.float64)
    sets = nonconf <= thresh_arr
    empty_mask = ~sets.any(axis=1)
    empty_rate = float(empty_mask.mean())
    if guarantee_nonempty and empty_mask.any():
        sets = sets.copy()
        top1 = probs[empty_mask].argmax(axis=1)
        sets[empty_mask, top1] = True
    return sets, empty_rate


def evaluate_conformal_oof(
    probs: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    is_transition: np.ndarray | None = None,
    *,
    alphas: tuple[float, ...] = (0.10, 0.05),
    stages: tuple[str, ...] = ("Wake", "N1", "N2", "N3", "REM"),
) -> dict:
    """Strictly out-of-fold split (marginal) and Mondrian conformal evaluation."""
    n = len(y)
    n_classes = len(stages)
    if probs.shape != (n, n_classes):
        raise ValueError(f"Expected probs shape ({n}, {n_classes}), got {probs.shape}")
    pred_top1 = probs.argmax(axis=1)
    fold_ids = sorted(np.unique(folds).tolist())
    results: dict[str, dict] = {}

    for alpha in alphas:
        target_cov = round(1.0 - alpha, 4)
        key = f"coverage_{int(round(target_cov * 100))}"
        alpha_entry: dict[str, dict] = {}
        for mode in ("marginal", "mondrian"):
            oof_sets = np.zeros((n, n_classes), dtype=bool)
            fold_records = []
            for f in fold_ids:
                cal_mask = folds != f
                test_mask = folds == f
                if mode == "marginal":
                    thr: float | np.ndarray = fit_split_conformal_threshold(
                        probs[cal_mask], y[cal_mask], alpha
                    )
                    thr_repr: float | list[float] = float(thr)
                else:
                    thr = fit_mondrian_conformal_thresholds(
                        probs[cal_mask], y[cal_mask], alpha, n_classes=n_classes
                    )
                    thr_repr = [float(v) for v in thr]
                sets_f, empty_f = conformal_prediction_sets(probs[test_mask], thr)
                oof_sets[test_mask] = sets_f
                cov_f = float(sets_f[np.arange(test_mask.sum()), y[test_mask]].mean())
                fold_records.append(
                    {
                        "fold": int(f),
                        "threshold": thr_repr,
                        "coverage": cov_f,
                        "mean_set_size": float(sets_f.sum(axis=1).mean()),
                        "empty_before_fallback_rate": empty_f,
                    }
                )

            covered = oof_sets[np.arange(n), y]
            sizes = oof_sets.sum(axis=1)
            singletons = sizes == 1
            multi = sizes > 1

            per_stage = {}
            for idx, stage in enumerate(stages):
                mask = y == idx
                per_stage[stage] = {
                    "epochs": int(mask.sum()),
                    "coverage": float(covered[mask].mean()),
                    "mean_set_size": float(sizes[mask].mean()),
                    "singleton_rate": float(singletons[mask].mean()),
                }

            mode_summary = {
                "target_coverage": target_cov,
                "empirical_coverage": float(covered.mean()),
                "mean_set_size": float(sizes.mean()),
                "median_set_size": float(np.median(sizes)),
                "singleton_rate": float(singletons.mean()),
                "singleton_accuracy": (
                    float((pred_top1[singletons] == y[singletons]).mean())
                    if singletons.any()
                    else None
                ),
                "ambiguous_rate": float(multi.mean()),
                "ambiguous_top1_accuracy": (
                    float((pred_top1[multi] == y[multi]).mean())
                    if multi.any()
                    else None
                ),
                "per_stage": per_stage,
                "folds": fold_records,
            }
            if is_transition is not None and is_transition.any() and (~is_transition).any():
                mode_summary["transition_epochs"] = {
                    "count": int(is_transition.sum()),
                    "coverage": float(covered[is_transition].mean()),
                    "mean_set_size": float(sizes[is_transition].mean()),
                    "singleton_rate": float(singletons[is_transition].mean()),
                }
                mode_summary["interior_epochs"] = {
                    "count": int((~is_transition).sum()),
                    "coverage": float(covered[~is_transition].mean()),
                    "mean_set_size": float(sizes[~is_transition].mean()),
                    "singleton_rate": float(singletons[~is_transition].mean()),
                }
            alpha_entry[mode] = mode_summary
        results[key] = alpha_entry
    return results


__all__ = [
    "CONCEPT_EPOCH_SIGNALS",
    "CONCEPT_NAMES",
    "CONCEPT_SLICES",
    "CONCEPT_STEP_SIGNALS",
    "ConceptBiGRUStagingNet",
    "ConceptTerms",
    "DEFAULT_CONCEPT_DIMS",
    "EMAWeights",
    "FEATURE_NAMES_79",
    "STEP_PER_EPOCH",
    "TOTAL_CONCEPT_DIM",
    "build_concept_layout",
    "calibrate_prior_power",
    "conformal_prediction_sets",
    "evaluate_conformal_oof",
    "fit_mondrian_conformal_thresholds",
    "fit_prior_power_oof",
    "fit_split_conformal_threshold",
    "smoothed_stage_loss",
]
