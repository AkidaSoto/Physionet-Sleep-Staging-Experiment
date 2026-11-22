"""Label-free, parameterized CinC staging measurements over existing caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping
import warnings

import numpy as np
import pandas as pd

from physionet_sleep.algorithms.eeg_quality import QUALITY_RULES, bad_seconds
from physionet_sleep.algorithms.eeg_tracks import EPS, WIN
from physionet_sleep.analysis.epoch_windows import robust_scale
from physionet_sleep.experiments.cached_staging_tuning import _block_mean, _local_reference
from physionet_sleep.experiments.staging_feature_waveform import EOG_VARIANTS, SLOW_WAVE_VARIANTS


@dataclass(frozen=True)
class FeatureFamily:
    slot: str
    contrast: tuple[int, int]
    guard: tuple[int, int] | None
    stratum: str
    options: Mapping[str, tuple]


NORMS = ("none", "median_iqr")
SCALE = (6, 30)
FAMILIES = {
    "sigma": FeatureFamily("spindle_bursts", (2, 1), (2, 3), "interior", {
        "reference": ("shared_flank", "disjoint_flank", "theta_beta_flank", "total", "absolute", "local"),
        "pooling": ("mean", "quantile", "occupancy"),
        "pool_seconds": SCALE, "norm": NORMS,
    }),
    "theta": FeatureFamily("c_rel_theta", (1, 0), None, "boundary", {
        "region": ("central", "occipital"), "reference": ("total", "alpha"),
        "compression": ("identity", "sqrt", "log"), "pooling": ("mean", "quantile"),
        "pool_seconds": SCALE, "norm": NORMS,
    }),
    "slow": FeatureFamily("topography_features.frontal_slow_wave_fraction", (3, 2), None, "interior", {
        "variant": SLOW_WAVE_VARIANTS,
    }),
    "eog": FeatureFamily("epoch_ratio_features.eog_rem_vs_noise_db", (4, 1), (4, 0), "interior", {
        "variant": EOG_VARIANTS,
    }),
    "lead": FeatureFamily("spindle_features.delta_power", (3, 2), (3, 4), "interior", {
        "relationship": ("frontal_central_ratio", "frontal_central_shared", "bilateral_frontal_shared"),
        "quality_gate": (True, False), "pooling": ("mean", "quantile"),
        "pool_seconds": SCALE, "norm": NORMS,
    }),
    "atonia": FeatureFamily("eye_movement_activity.eye_movement_baseline", (4, 1), (4, 0), "interior", {
        "reference_percentile": (15, 25, 35), "radius_minutes": (3, 5, 10, 20),
        "clean_chin_only": (True, False),
    }),
}

ANCHORS = {
    "sigma": {"family": "sigma", "reference": "disjoint_flank",
              "pooling": "occupancy", "pool_seconds": 30, "norm": "median_iqr", "threshold_mad": 2.0},
    "theta": {"family": "theta", "region": "occipital", "reference": "total", "compression": "identity",
              "pooling": "mean", "pool_seconds": 30, "norm": "median_iqr"},
    "slow": {"family": "slow", "variant": "slow_wave_60_best_clean"},
    "eog": {"family": "eog", "variant": "eog_short_6s_q90_unguarded"},
    "lead": {"family": "lead", "relationship": "frontal_central_shared", "quality_gate": True,
             "pooling": "mean", "pool_seconds": 6, "norm": "none"},
    "atonia": {"family": "atonia", "reference_percentile": 25, "radius_minutes": 5, "clean_chin_only": False},
}


def sample_recipe(family: str, suggest: Callable[[str, tuple], Any]) -> dict:
    """Sample only active parameters, using family-qualified parameter names."""
    options = FAMILIES[family].options
    recipe = {"family": family, **{name: suggest(f"{family}.{name}", values) for name, values in options.items()}}
    if recipe.get("reference") == "local":
        recipe["reference_seconds"] = suggest(f"{family}.reference_seconds", (60, 180, 300, 600))
    if recipe.get("pooling") == "quantile":
        recipe["quantile"] = suggest(f"{family}.quantile", (0.75, 0.9))
    if recipe.get("pooling") == "occupancy":
        recipe["threshold_mad"] = suggest(f"{family}.threshold_mad", (1.0, 1.5, 2.0, 2.5))
    elif family == "sigma":
        recipe["compression"] = suggest("sigma.compression", ("identity", "sqrt", "log"))
    return recipe


def validate_recipe(recipe: Mapping) -> None:
    family = recipe.get("family")
    if family not in FAMILIES:
        raise ValueError(f"Unknown feature family {family!r}")

    def read(name: str, options: tuple):
        field = name.split(".", 1)[1]
        value = recipe.get(field)
        if value not in options:
            raise ValueError(f"Invalid {name} value {value!r}; expected one of {options!r}")
        return value

    expected = sample_recipe(family, read)
    if set(recipe) != set(expected):
        raise ValueError(f"Inactive or unknown parameters in {family}: {sorted(set(recipe) - set(expected))}")


def recipe_key(recipe: Mapping) -> str:
    """Stable filename-safe key for a fully specified feature recipe."""
    validate_recipe(recipe)
    canonical = json.dumps(dict(recipe), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"{recipe['family']}-{hashlib.sha256(canonical.encode()).hexdigest()[:20]}"


def feature_metadata(recipe: Mapping) -> dict:
    validate_recipe(recipe)
    family = recipe["family"]
    if family == "eog":
        units = "dB"
    elif family in ("atonia", "slow") or recipe.get("pooling") == "occupancy":
        units = "fraction"
    elif family == "lead" and recipe["relationship"] == "frontal_central_ratio":
        units = "log power-share ratio"
    elif family == "sigma" and recipe["reference"] == "absolute":
        units = "uV^2"
    else:
        units = "ratio"
    norm = recipe.get("norm", "none")
    compression = "identity" if recipe.get("pooling") == "occupancy" else recipe.get("compression", "identity")
    transformed_units = {"identity": units, "sqrt": f"sqrt({units})", "log": f"log({units})"}[compression]
    return {"slot": FAMILIES[family].slot, "units": transformed_units if norm == "none" else "IQR units",
            "physical_units": units, "normalization": norm, "parameters": dict(recipe),
            "pool_seconds": recipe.get("pool_seconds", 30), "source": {
                "sigma": "tracks1s", "theta": "tracks1s + quality1s", "lead": "quality1s",
                "slow": "archived calibrated waveform features", "eog": "archived calibrated waveform features",
                "atonia": "rows6 + quality1s when enabled",
            }[family]}


_BANDS = ["c_rel_delta", "c_rel_theta", "c_rel_alpha", "c_rel_sigma", "c_rel_beta"]
_TRACK_COLUMNS = ["second", *_BANDS, "c_log_power"]
_QUALITY_COLUMNS = ["second", "EMG_flat", *(f"{lead}_{k}" for lead in ("F3", "F4", "C3", "C4")
                     for k in ("delta", "total", *QUALITY_RULES)),
                    *(f"{lead}_{k}" for lead in ("O1", "O2") for k in ("theta", "alpha", "total"))]
_CHIN = "epoch_ratio_features.emg_floor_db"
_SQUARED_VOLTAGE_SCALE = 2 / (WIN * np.square(np.hanning(WIN)).sum())


def _as_float(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > EPS)


def _pooled(values: np.ndarray, recipe: Mapping) -> np.ndarray:
    width = recipe["pool_seconds"]
    pooling = recipe["pooling"]
    if pooling == "quantile":
        blocks = np.pad(values, (0, -len(values) % width), constant_values=np.nan).reshape(-1, width)
        counts = np.isfinite(blocks).sum(axis=1)
        ordered = np.sort(np.where(np.isfinite(blocks), blocks, np.nan), axis=1)
        rank = np.maximum(counts - 1, 0) * recipe["quantile"]
        lower, upper = np.floor(rank).astype(int), np.ceil(rank).astype(int)
        index = np.arange(len(rank))
        a, b = ordered[index, lower], ordered[index, upper]
        pooled = a + (b - a) * (rank - lower)
    else:
        if pooling == "occupancy" and np.isfinite(values).any():
            median = np.nanmedian(values)
            cutoff = median + recipe["threshold_mad"] * 1.4826 * np.nanmedian(np.abs(values - median))
            values = np.where(np.isfinite(values), values > cutoff, np.nan)
        pooled = _block_mean(values, width)
    return np.repeat(pooled, width)[:len(values)]


class CachedRecipeNight:
    """Read projected source columns lazily and hold only the current night's sources."""

    def __init__(self, cache_dir: Path, subject_id: str, waveform_cache: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.subject_id = subject_id
        self.waveform_cache = Path(waveform_cache) if waveform_cache is not None else (
            self.cache_dir.parent / "exploration" / "cinc-feature-tuning" / "waveform_cache"
        )
        self._frames: dict[str, pd.DataFrame] = {}

    def _grid(self, kind: str, columns: list[str]) -> pd.DataFrame:
        if kind not in self._frames:
            frame = pd.read_parquet(self.cache_dir / kind / f"{self.subject_id}.parquet", columns=columns)
            clock = frame["second"].to_numpy()
            if not len(clock) or not np.array_equal(clock, np.arange(len(clock))):
                raise ValueError(f"Broken {kind} clock for {self.subject_id}")
            self._frames[kind] = frame
        return self._frames[kind]

    @property
    def tracks(self) -> pd.DataFrame:
        return self._grid("tracks1s", _TRACK_COLUMNS)

    @property
    def quality(self) -> pd.DataFrame:
        quality = self._grid("quality1s", _QUALITY_COLUMNS)
        if len(quality) != len(self.tracks):
            raise ValueError(f"Quality and tracks grids differ for {self.subject_id}")
        return quality

    def _sigma(self, recipe: Mapping) -> np.ndarray:
        delta, theta, alpha, sigma, beta = _as_float(self.tracks[_BANDS]).T
        ref = recipe["reference"]
        power = np.power(10., _as_float(self.tracks.c_log_power))
        if ref == "absolute":
            return sigma * power * _SQUARED_VOLTAGE_SCALE
        if ref == "total":
            return sigma
        if ref == "local":
            background, _ = _local_reference(sigma, recipe["reference_seconds"])
            return _ratio(sigma, background)
        if ref == "theta_beta_flank":
            return _ratio(sigma, theta + beta)
        if ref == "disjoint_flank":
            # The pooled 8–12 Hz and 11–16 Hz bands share 11–12 Hz. Their
            # excess over the exhaustive 0.5–30 Hz partition recovers it.
            overlap = np.clip(delta + theta + alpha + sigma + beta - (1 - EPS / power), 0, np.minimum(alpha, sigma))
            alpha = alpha - overlap
        return _ratio(sigma, alpha + beta)

    def _theta(self, recipe: Mapping) -> np.ndarray:
        if recipe["region"] == "central":
            theta = _as_float(self.tracks.c_rel_theta)
            return theta if recipe["reference"] == "total" else _ratio(theta, _as_float(self.tracks.c_rel_alpha))
        q = self.quality
        ref = recipe["reference"]
        return _ratio(_as_float(q.O1_theta) + _as_float(q.O2_theta), _as_float(q[f"O1_{ref}"]) + _as_float(q[f"O2_{ref}"]))

    def _lead(self, recipe: Mapping) -> np.ndarray:
        q = self.quality
        shares = {}
        for lead in ("F3", "F4", "C3", "C4"):
            values = _ratio(_as_float(q[f"{lead}_delta"]), _as_float(q[f"{lead}_total"]))
            if recipe["quality_gate"]:
                values[bad_seconds(q, lead)] = np.nan
            shares[lead] = values
        relation = recipe["relationship"]
        if relation == "bilateral_frontal_shared":
            return np.minimum(shares["F3"], shares["F4"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            frontal = np.nanmean(np.stack([shares["F3"], shares["F4"]]), axis=0)
            central = np.nanmean(np.stack([shares["C3"], shares["C4"]]), axis=0)
        if relation == "frontal_central_ratio":
            return np.log(np.maximum(frontal, EPS)) - np.log(np.maximum(central, EPS))
        return np.minimum(frontal, central)

    def _archived(self, column: str) -> np.ndarray:
        if "waveform" not in self._frames:
            all_columns = [*SLOW_WAVE_VARIANTS, *EOG_VARIANTS]
            frame = pd.read_parquet(self.waveform_cache / f"{self.subject_id}.parquet", columns=all_columns)
            if not np.array_equal(frame.index.to_numpy(), np.arange(len(frame))):
                raise ValueError(f"Broken archived waveform clock for {self.subject_id}")
            self._frames["waveform"] = frame
        frames = np.repeat(_as_float(self._frames["waveform"][column]), 30)
        n = len(self.tracks)
        return np.pad(frames[:n], (0, max(0, n - len(frames))), constant_values=np.nan)

    def _atonia(self, recipe: Mapping) -> np.ndarray:
        if "rows6" not in self._frames:
            rows = pd.read_parquet(self.cache_dir / "rows6" / f"{self.subject_id}.parquet", columns=["slot", _CHIN])
            if not rows.slot.is_unique or rows.slot.min() < 0 or not pd.api.types.is_integer_dtype(rows.slot):
                raise ValueError(f"Broken rows6 clock for {self.subject_id}")
            self._frames["rows6"] = rows
        rows = self._frames["rows6"]
        n = len(self.tracks)
        slots = np.arange(-(-n // 6))
        values = np.repeat(rows.set_index("slot")[_CHIN].reindex(slots).to_numpy(dtype=np.float64), 6)[:n]
        if recipe["clean_chin_only"]:
            values[bad_seconds(self.quality, "EMG")] = np.nan
        chin = _block_mean(values, 30)
        eligible = np.isfinite(chin)
        low = eligible & (chin <= np.nanpercentile(chin, recipe["reference_percentile"])) if eligible.any() else eligible
        t = np.arange(len(chin))
        reach = recipe["radius_minutes"] * 2
        # Exclude the focal epoch and its two immediate neighbors on each side.
        before, after = np.maximum(0, t - 2), np.minimum(len(chin), t + 3)
        start, end = np.maximum(0, t - reach), np.minimum(len(chin), t + reach + 1)
        def surround(x: np.ndarray) -> np.ndarray:
            count = np.r_[0, np.cumsum(x)]
            return count[before] - count[start] + count[end] - count[after]

        total = surround(eligible)
        fraction = np.divide(surround(low), total, out=np.full(len(chin), np.nan), where=total > 0)
        return np.repeat(fraction, 30)[:n]

    def compute(self, recipe: Mapping, *, normalize: bool = True) -> np.ndarray:
        """Produce a dense float32 vector, with coarse pools held on the one-second grid."""
        validate_recipe(recipe)
        family = recipe["family"]
        if family in ("eog", "slow"):
            return self._archived(recipe["variant"]).astype(np.float32)
        if family == "atonia":
            return self._atonia(recipe).astype(np.float32)
        values = {"theta": self._theta, "sigma": self._sigma, "lead": self._lead}[family](recipe)
        # For occupancy, threshold the native quantity: a monotone compression
        # would change the distance to the robust cutoff rather than count the same event.
        compression = "identity" if recipe["pooling"] == "occupancy" else recipe.get("compression", "identity")
        if compression == "sqrt":
            values = np.sqrt(values)
        if compression == "log":
            values = np.log(np.maximum(values, EPS))
        values = _pooled(values, recipe).astype(np.float32)
        if normalize and recipe.get("norm", "none") == "median_iqr":
            values = robust_scale(values[:, None], [family])[:, 0]
        return values
