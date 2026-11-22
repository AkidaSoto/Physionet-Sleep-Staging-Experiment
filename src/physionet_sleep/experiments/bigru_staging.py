"""Whole-night bidirectional gated GRU staging model and unified feature helpers (#75)."""

from __future__ import annotations

from math import log
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.nn import functional as F

from physionet_sleep.algorithms.eeg_tracks import EPS, FS, band_power, power_spectra
from physionet_sleep.analysis.epoch_windows import EPOCH_SECONDS, ROW_SECONDS
from physionet_sleep.experiments.complementary_n2_events import (
    N2_EVENT_COLUMNS,
    _recent_support,
)
from physionet_sleep.experiments.complementary_slow_waves import SLOW_WAVE_PANEL_COLUMNS
from physionet_sleep.experiments.gated_staging import StagingTerms


BASE_STEP_TRACKS = (
    "c_rel_theta",
    "c_rel_alpha",
    "c_rel_beta",
    "c_log_power",
    "o_rel_alpha",
    "eog_high",
)
EXTRA_STEP_TRACKS = ("c_rel_sigma", "sigma_ratio", "o_rel_theta")
STEP_TRACKS = (*BASE_STEP_TRACKS, *EXTRA_STEP_TRACKS)
STEP_PER_EPOCH = EPOCH_SECONDS // ROW_SECONDS

COMPLEMENTARY_MARKERS = (*SLOW_WAVE_PANEL_COLUMNS, *N2_EVENT_COLUMNS)
SPLIT_SUPPORT_COLUMNS = ("recent_k_support", "recent_spindle_support")
NIGHT_HISTORY_COLUMNS = (
    "elapsed_hours",
    "elapsed_squared",
    "fraction_elapsed",
    "slow_burden_vs_clock",
    "slow_coverage_prior_30min",
    "atonia_fraction_10min",
    "nearby_eye_atonia",
)
OCCIPITAL_EPOCH_COLUMN = ("o_theta_30s",)
EPOCH_FEATURE_COLUMNS = (
    *COMPLEMENTARY_MARKERS,
    *SPLIT_SUPPORT_COLUMNS,
    *NIGHT_HISTORY_COLUMNS,
    *OCCIPITAL_EPOCH_COLUMN,
)

SLOW_COL = "topography_features.frontal_slow_wave_fraction"
CHIN_COL = "epoch_ratio_features.emg_floor_db"
EYE_COL = "epoch_ratio_features.eog_rem_vs_noise_db"
NIGHT_HISTORY_SOURCE_COLUMNS = (SLOW_COL, CHIN_COL, EYE_COL)

DEFAULT_STEP_DIM = 21 * STEP_PER_EPOCH
DEFAULT_EPOCH_DIM = len(EPOCH_FEATURE_COLUMNS)


def extract_occipital_theta(
    o1_uv: np.ndarray | None,
    o2_uv: np.ndarray | None,
) -> pd.DataFrame:
    """Compute 1 Hz occipital relative theta (4-8 Hz / 0.5-30 Hz) on (O1-M2 + O2-M1) / 2."""
    if o1_uv is None and o2_uv is None:
        raise ValueError("At least one occipital channel is required")
    if o1_uv is None:
        o1_uv = o2_uv
    if o2_uv is None:
        o2_uv = o1_uv
    o1 = np.asarray(o1_uv, dtype=np.float32)
    o2 = np.asarray(o2_uv, dtype=np.float32)
    n_seconds = min(len(o1), len(o2)) // FS
    if n_seconds <= 0:
        raise ValueError("Occipital leads contain fewer than one complete second")
    n_samples = n_seconds * FS
    o1_seg = o1[:n_samples]
    o2_seg = o2[:n_samples]
    o1_ok = np.isfinite(o1_seg)
    o2_ok = np.isfinite(o2_seg)
    both_ok = o1_ok & o2_ok
    combined = np.where(
        both_ok,
        0.5 * (o1_seg + o2_seg),
        np.where(o1_ok, o1_seg, np.where(o2_ok, o2_seg, 0.0)),
    ).astype(np.float32)
    spectra = power_spectra(combined, n_seconds)
    total = band_power(spectra, 0.5, 30.0)
    theta = band_power(spectra, 4.0, 8.0)
    ratio = theta / (total + EPS)
    sec_ok = (o1_ok | o2_ok).reshape(n_seconds, FS).all(axis=1) & (total > 0)
    values = np.where(sec_ok, ratio, np.nan).astype(np.float32)
    return pd.DataFrame(
        {"second": np.arange(n_seconds, dtype=np.int64), "o_rel_theta": values}
    )


def compute_sigma_ratio(tracks: pd.DataFrame) -> np.ndarray:
    """Compute continuous 1 Hz spindle ratio c_rel_sigma / (c_rel_alpha + c_rel_beta + 1e-6)."""
    sigma = tracks["c_rel_sigma"].to_numpy(dtype=np.float32)
    alpha = tracks["c_rel_alpha"].to_numpy(dtype=np.float32)
    beta = tracks["c_rel_beta"].to_numpy(dtype=np.float32)
    return (sigma / (alpha + beta + 1e-6)).astype(np.float32)


def block_epoch_mean(values_1s: np.ndarray, n_epochs: int) -> np.ndarray:
    """Average a 1 Hz signal onto complete and trailing partial 30 s epochs."""
    arr = np.asarray(values_1s, dtype=np.float32)
    target = n_epochs * EPOCH_SECONDS
    if len(arr) >= target:
        trimmed = arr[:target]
    else:
        trimmed = np.pad(arr, (0, target - len(arr)), constant_values=np.nan)
    blocks = trimmed.reshape(n_epochs, EPOCH_SECONDS)
    finite = np.isfinite(blocks)
    counts = finite.sum(axis=1)
    sums = np.where(finite, blocks, 0.0).sum(axis=1, dtype=np.float64)
    return np.divide(
        sums,
        counts,
        out=np.full(n_epochs, np.nan, dtype=np.float64),
        where=counts > 0,
    ).astype(np.float32)


def compute_split_support(
    kc_events: np.ndarray, spindle_counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal 4-epoch decayed K-like support and spindle-burst support (#67)."""
    events = np.asarray(kc_events, dtype=np.float32)
    spindles = np.asarray(spindle_counts, dtype=np.float32)
    if events.shape != spindles.shape or events.ndim != 1:
        raise ValueError("K-like events and spindle counts must share a 1D epoch shape")
    recent_k = _recent_support(events, np.zeros_like(events)).astype(np.float32)
    recent_spindle = _recent_support(np.zeros_like(spindles), spindles).astype(np.float32)
    return recent_k, recent_spindle


def compute_night_history(rows6: pd.DataFrame, n_seconds: int) -> np.ndarray:
    """Compute #51's 7 label-free night-clock and macro-physiology features on 30 s epochs."""
    n = -(-n_seconds // EPOCH_SECONDS)
    observed = np.full((n * STEP_PER_EPOCH, 3), np.nan, dtype=np.float64)
    indexed = rows6.loc[rows6["slot"].between(0, len(observed) - 1)]
    observed[indexed["slot"].to_numpy()] = indexed[
        list(NIGHT_HISTORY_SOURCE_COLUMNS)
    ].to_numpy(dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        slow, chin, eye = np.nanmean(
            observed.reshape(n, STEP_PER_EPOCH, 3), axis=1
        ).T
    t = np.arange(n)
    valid_chin = np.isfinite(chin)
    valid_eye = np.isfinite(eye)
    q_chin = np.nanquantile(chin, 0.25) if valid_chin.any() else -np.inf
    q_eye = np.nanquantile(eye, 0.75) if valid_eye.any() else np.inf
    atonia = valid_chin & (chin <= q_chin)
    anchor = np.flatnonzero(atonia & valid_eye & (eye > q_eye))
    end_before = np.maximum(0, t - 2)
    start_after = np.minimum(n, t + 3)

    slow_sum = np.r_[0.0, np.nancumsum(np.clip(slow, 0.0, 1.0))]
    slow_n = np.r_[0, np.cumsum(np.isfinite(slow))]
    since = np.maximum(0, end_before - 60)
    count = slow_n[end_before] - slow_n[since]
    prior_slow = np.divide(
        slow_sum[end_before] - slow_sum[since],
        count,
        out=np.zeros(n, dtype=np.float64),
        where=count > 0,
    )
    burden = (
        slow_sum[end_before] / slow_sum[-1] - end_before / n
        if slow_sum[-1] > 0
        else np.zeros(n, dtype=np.float64)
    )

    chin_end, chin_start = np.minimum(n, t + 11), np.maximum(0, t - 10)
    low_n = np.r_[0, np.cumsum(atonia)]
    total_n = np.r_[0, np.cumsum(valid_chin)]
    left = low_n[end_before] - low_n[chin_start]
    right = low_n[chin_end] - low_n[start_after]
    count_chin = (
        total_n[end_before]
        - total_n[chin_start]
        + total_n[chin_end]
        - total_n[start_after]
    )
    persistence = np.divide(
        left + right,
        count_chin,
        out=np.zeros(n, dtype=np.float64),
        where=count_chin > 0,
    )

    prev = np.r_[-10 * n, anchor]
    nex = np.r_[anchor, 10 * n]
    left_gap = t - prev[np.searchsorted(anchor, end_before, side="left")]
    right_gap = nex[np.searchsorted(anchor, start_after, side="left")] - t
    close_anchor = 1.0 - np.minimum(np.minimum(left_gap, right_gap), 60.0) / 60.0
    hours = t / 120.0
    return np.column_stack(
        [
            hours,
            hours**2,
            t / max(1, n - 1),
            burden,
            prior_slow,
            persistence,
            close_anchor,
        ]
    ).astype(np.float32)


class UnifiedBiGRUStagingNet(nn.Module):
    """Whole-night bidirectional gated GRU with step-level and epoch-level stem projections."""

    def __init__(
        self,
        step_dim: int = DEFAULT_STEP_DIM,
        epoch_dim: int = DEFAULT_EPOCH_DIM,
        width: int = 96,
        dropout: float = 0.15,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if step_dim < 1:
            raise ValueError("step_dim must be positive")
        if epoch_dim < 0:
            raise ValueError("epoch_dim must be non-negative")
        self.step_dim = step_dim
        self.epoch_dim = epoch_dim
        self.width = width

        self.step_proj = nn.Linear(step_dim, width)
        self.epoch_proj = (
            nn.Linear(epoch_dim, width, bias=False) if epoch_dim > 0 else None
        )
        if self.epoch_proj is not None:
            nn.init.zeros_(self.epoch_proj.weight)
        self.stem_norm = nn.Sequential(nn.LayerNorm(width), nn.GELU())

        self.local_classifier = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 5),
        )
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.past_rnn = nn.GRU(
            width, width, num_layers=num_layers, batch_first=True, dropout=rnn_dropout
        )
        self.future_rnn = nn.GRU(
            width, width, num_layers=num_layers, batch_first=True, dropout=rnn_dropout
        )
        self.past_classifier = self._context_head(width, dropout)
        self.future_classifier = self._context_head(width, dropout)
        for head in (self.past_classifier, self.future_classifier):
            nn.init.zeros_(head[-1].weight)

        self.gates = nn.Linear(width, 10)
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

    def forward(
        self,
        x: torch.Tensor,
        recorded: torch.Tensor,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | StagingTerms:
        expected_dim = self.step_dim + self.epoch_dim
        if x.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected feature width {expected_dim}, received {x.shape[-1]}"
            )
        valid = recorded.unsqueeze(-1)
        x_clean = x.masked_fill(~valid, 0)
        pre = self.step_proj(x_clean[..., : self.step_dim])
        if self.epoch_proj is not None:
            pre = pre + self.epoch_proj(x_clean[..., self.step_dim :])
        hidden = self.stem_norm(pre).masked_fill(~valid, 0)
        local = self.local_classifier(hidden).masked_fill(~valid, 0)

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

            # Reverse only each night's valid prefix so padded batch tails never
            # shift the GRU's starting bias state before the final recorded epoch.
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
            future_available = valid & (future_count > 0).unsqueeze(-1)
            future_raw = self.future_classifier(future_hidden).masked_fill(
                ~future_available, 0
            )

        past_gate, future_gate = self.gates(hidden).sigmoid().chunk(2, dim=-1)
        past = past_raw * past_gate
        future = future_raw * future_gate
        logits = local + past + future
        if return_components:
            return StagingTerms(logits, local, past, future, past_gate, future_gate)
        return logits


def calibrate_prior_power(
    probabilities: np.ndarray, prior: np.ndarray, exponent: float
) -> np.ndarray:
    """Apply normalized prior-power reweighting p * prior^a in log space."""
    log_p = np.log(np.clip(probabilities, 1e-12, None))
    log_prior = np.log(np.clip(prior, 1e-12, None))
    return softmax(log_p + exponent * log_prior[None, :], axis=1).astype(np.float32)


def fit_prior_power_oof(
    probabilities: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Fit p * prior^a strictly on the 4 outer-training folds and apply out-of-fold."""
    if grid is None:
        grid = np.linspace(-0.25, 1.0, 251)
    n_classes = probabilities.shape[1]
    labels = np.arange(n_classes)
    calibrated = np.empty_like(probabilities, dtype=np.float32)
    details = []
    for fold in sorted(np.unique(folds).tolist()):
        train_mask = folds != fold
        test_mask = folds == fold
        if not train_mask.any() or not test_mask.any():
            raise ValueError(f"Fold {fold} requires non-empty train and test masks")
        counts = np.bincount(y[train_mask], minlength=n_classes).astype(np.float64)
        prior = counts / counts.sum()
        log_p_train = np.log(np.clip(probabilities[train_mask], 1e-12, None))
        log_prior = np.log(np.clip(prior, 1e-12, None))
        y_train = y[train_mask]
        best_score = -1.0
        best_a = 0.0
        for a in grid:
            pred = (log_p_train + float(a) * log_prior[None, :]).argmax(axis=1)
            score = float(
                f1_score(y_train, pred, labels=labels, average="macro", zero_division=0)
            )
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and abs(float(a)) < abs(best_a)
            ):
                best_score = score
                best_a = float(a)
        calibrated[test_mask] = calibrate_prior_power(
            probabilities[test_mask], prior, best_a
        )
        details.append(
            {
                "fold": int(fold),
                "exponent": best_a,
                "outer_train_macro_f1": best_score,
                "outer_train_prior": prior.tolist(),
            }
        )
    return calibrated, details
