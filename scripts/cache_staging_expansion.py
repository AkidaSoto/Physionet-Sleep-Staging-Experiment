"""Prepare the fixed Boston training-only expansion with the archived staging extractors."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import replace
from functools import wraps
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physionet_sleep.algorithms.eeg_quality import channel_quality
from physionet_sleep.algorithms.eeg_tracks import BANDS, EPS, FS, SPECTRAL_CHANNELS, band_power, power_spectra
from physionet_sleep.analysis.datasets import DATASETS
from physionet_sleep.analysis.epoch_windows import (
    DERIVED_FEATURES,
    TARGET,
    subject_epochs,
    subject_rows6,
)
from physionet_sleep.core.types import SignalRecord
from physionet_sleep.experiments.waveform_data import WaveformStore
from physionet_sleep.io.cinc2018 import CHANNEL_MAP, EEG_STACKS, resolve_cinc2018_record
from physionet_sleep.recipe.algo_store import ParquetAlgoStore
from physionet_sleep.studies.cinc2018 import CinC2018StagingStudyDef

RAW = ROOT / "data/raw/cinc2018"
BASE = ROOT / "artifacts/staging_window_model/cinc2018"
ARCHIVE_WINDOW = ROOT / "artifacts/cinc2018_window_cache"
EXPLORATION = ROOT / "artifacts/exploration"
OUT = EXPLORATION / "cinc-all-additional-training"
WINDOW = OUT / "window_cache"
MARKERS = OUT / "feature_cache"
SCORED = OUT / "scored_epochs_by_subject"
SCRATCH = OUT / "scratch"
KEYS = ["subject_id", "epoch_index"]
FAMILIES = ("rows6", "tracks1s")
GIB = 2**30
MIB = 2**20
FRONTAL_LEADS = ("F3-M2", "F4-M1")
TRACK_LEADS = tuple(name for name in SPECTRAL_CHANNELS if name != "Chin1-Chin2")


class CompactNight:
    """Calibrate only the archived input channels from the interleaved CinC record."""

    def __init__(self, subject: str) -> None:
        import wfdb

        paths = resolve_cinc2018_record(RAW, subject)
        source = paths.signal_path
        if source is None:
            raise FileNotFoundError(f"{subject}: missing waveform")
        header = wfdb.rdheader(str(paths.header_path.with_suffix("")))
        names = {*SPECTRAL_CHANNELS, "ECG"}
        if not (
            header.fs == FS
            and header.sig_len > 0
            and names.issubset(header.sig_name)
            and set(header.file_name) == {source.name}
            and set(header.fmt) == {"16"}
            and set(header.byte_offset) == {24}
            and set(header.samps_per_frame) == {1}
            and set(header.skew).issubset({None, 0})
            and source.stat().st_size == 24 + 2 * header.sig_len * header.n_sig
        ):
            raise ValueError(f"{subject}: expected a complete interleaved 200 Hz CinC waveform")
        self.header = header
        self.columns = {name: header.sig_name.index(name) for name in names}
        gains = np.asarray([header.adc_gain[index] for index in self.columns.values()])
        if not np.isfinite(gains).all() or (gains <= 0).any():
            raise ValueError(f"{subject}: invalid waveform calibration gain")
        self.samples = np.memmap(
            source, mode="r", dtype="<i2", offset=24, shape=(header.sig_len, header.n_sig)
        )

    def physical(self, name: str) -> np.ndarray:
        index = self.columns[name]
        digital = self.samples[:, index]
        # Match WFDB's original native-unit float64 DAC before any float32 cast.
        values = digital.astype(np.float64)
        values -= self.header.baseline[index]
        values /= self.header.adc_gain[index]
        values[digital == np.iinfo(np.int16).min] = np.nan
        return values

    def recipe_record(self) -> SignalRecord:
        source_names = {key: name for name, key in CHANNEL_MAP.items()}
        result = SignalRecord()
        for stack, members in EEG_STACKS.items():
            names = [source_names[key] for key in members]
            values = np.empty((self.header.sig_len, len(names)), dtype=np.float64)
            for index, name in enumerate(names):
                values[:, index] = self.physical(name)
            result.add_channel(stack, values, FS, units=self.header.units[self.columns[names[0]]])
        for name in ("E1-M2", "Chin1-Chin2", "ECG"):
            result.add_channel(
                CHANNEL_MAP[name], self.physical(name), FS, units=self.header.units[self.columns[name]]
            )
        return result


class CompactStudy(CinC2018StagingStudyDef):
    def __init__(self, subject: str, cache: Path, night: CompactNight) -> None:
        super().__init__(subjects=[subject], root_dir=RAW, cache_dir=cache)
        self.night = night

    def load_fn(self, subject_id: str, req: str, meta=None):
        if req == "signal_record":
            return self.night.recipe_record()
        return super().load_fn(subject_id, req, meta)


def listed(path: Path, count: int) -> list[str]:
    subjects = path.read_text(encoding="utf-8").split()
    if len(subjects) != count or len(set(subjects)) != count:
        raise ValueError(f"{path}: expected {count} distinct subjects")
    if any(re.fullmatch(r"tr[0-9]{2}-[0-9]{4}", subject) is None for subject in subjects):
        raise ValueError(f"{path}: invalid CinC record ID")
    return subjects


def development_and_holdout() -> tuple[set[str], set[str]]:
    dev = set(listed(BASE / "dev_subjects.txt", 320))
    holdout = set(listed(RAW / "selected_records_holdout.txt", 200))
    ledger = set(pd.read_csv(BASE / "holdout_subjects.csv", usecols=["record_id"]).record_id)
    if dev & holdout or ledger != holdout:
        raise ValueError("Independent Boston holdout manifests disagree")
    return dev, holdout


def expansion_manifest(dev: set[str], holdout: set[str]) -> dict[str, int]:
    subjects = listed(OUT / "subjects.txt", 474)
    observed = {header.stem for header in RAW.glob("tr*/*.hea")}
    if set(subjects) != observed - dev - holdout:
        raise ValueError("Expansion manifest differs from the unused Boston records")
    sizes = pd.read_csv(OUT / "waveform_manifest.csv")
    if sizes.subject_id.tolist() != subjects or (sizes.expected_bytes <= 24).any():
        raise ValueError("Waveform sizes and expansion IDs disagree")
    return dict(zip(subjects, sizes.expected_bytes, strict=True))


def parquet(frame: pd.DataFrame, path: Path, *, index: bool | None = False) -> None:
    pending = path.with_suffix(".part")
    frame.to_parquet(pending, index=index)
    pending.replace(path)


def peak_rss_mib() -> int:
    if os.name != "nt":
        from resource import RUSAGE_SELF, getrusage

        scale = 2**20 if sys.platform == "darwin" else 2**10
        return round(getrusage(RUSAGE_SELF).ru_maxrss / scale)

    import ctypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_ulong),
            ("faults", ctypes.c_ulong),
            ("peak_working_set", ctypes.c_size_t),
            ("remaining_counters", ctypes.c_size_t * 7),
        ]

    counters = Counters()
    counters.size = ctypes.sizeof(counters)
    get_counters = ctypes.windll.kernel32.K32GetProcessMemoryInfo
    get_counters.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not get_counters(ctypes.c_void_p(-1), ctypes.byref(counters), counters.size):
        raise ctypes.WinError()
    return round(counters.peak_working_set / 2**20)


def available_memory() -> int:
    if os.name != "nt":
        if "SC_AVPHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return sys.maxsize

    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("physical", ctypes.c_uint64),
            ("available", ctypes.c_uint64),
            ("remaining", ctypes.c_uint64 * 5),
        ]

    status = MemoryStatus()
    status.size = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return status.available


def estimated_peak(size: int) -> int:
    # A 24-byte header followed by 13 interleaved int16 channels. Large-prime
    # lengths measured higher during whole-night FFTs, so reserve more space.
    samples = (size - 24) // 26
    largest = samples
    divisor = 2
    while divisor * divisor <= largest:
        if largest % divisor == 0:
            largest //= divisor
        else:
            divisor += 1
    factor = 16 if largest * largest > samples else 9
    return 300 * MIB + factor * size


def timed_step(extract, name: str):
    @wraps(extract)
    def run(inputs: dict, params: dict) -> None:
        before = peak_rss_mib()
        started = monotonic()
        extract(inputs, params)
        print(f"{name}: {monotonic() - started:.1f}s, peak RSS {before} to {peak_rss_mib()} MiB", flush=True)

    return run


def stage_recipe(subject: str, cache: Path, columns: list[str], night: CompactNight, *, smoke: bool) -> None:
    from physionet_sleep.experiments.staging_spindle_inputs import StagingSpindleInputs

    study = CompactStudy(subject, cache, night)
    # Retain the stage-label dropout flag alongside the model's signal generators.
    compiler = study.recipe[-1]
    if compiler.algo_name() != "feature_table":
        raise ValueError("The production recipe no longer ends with its feature table")
    needed = {name.split(".", 1)[0] for name in columns} | {"signal_quality"}
    selected = []
    for step in reversed(study.recipe[:-1]):
        if step.algo_name() in needed:
            selected.append(step)
            needed.update(step.dependency_names)
    compiler.dependency_names = tuple(step.algo_name() for step in reversed(selected))
    if set(compiler.dependency_names) != needed:
        raise ValueError(f"Production recipe is missing dependencies: {needed - set(compiler.dependency_names)}")
    study.recipe = [*reversed(selected), compiler]
    spindle = next(step for step in study.recipe if step.algo_name() == "spindle_features")
    spindle.algorithm = StagingSpindleInputs(signal_key=spindle.algorithm.signal_key)
    ratio = next(step for step in study.recipe if step.algo_name() == "epoch_ratio_features")
    ratio.algorithm.include_emg_baseline = "epoch_ratio_features.emg_baseline_floor_db" in columns
    if smoke:
        for step in study.recipe:
            step.extract = timed_step(step.extract, step.algo_name())
    result = study.run_build({"no_save": True, "outputs": ["final__feature_table"]})
    if result["subjects"] != [subject]:
        raise ValueError(f"{subject}: production staging recipe failed")
    store = ParquetAlgoStore()
    table = result["cache"][0]["final__feature_table"]
    store.save(str(cache / subject), "final__feature_table", table)


def extract_seconds(night: CompactNight, tracks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # The archived periodogram and quality calculations both use float32 inputs.
    signals = {name: night.physical(name).astype(np.float32) for name in [*TRACK_LEADS, "ECG"]}
    n = len(signals["C3-M2"]) // FS
    spectra = {name: power_spectra(signals[name], n) for name in TRACK_LEADS}
    central = (spectra["C3-M2"] + spectra["C4-M1"]) / 2
    occipital = (spectra["O1-M2"] + spectra["O2-M1"]) / 2
    total = band_power(central, 0.5, 30) + EPS
    values = {f"c_rel_{name}": band_power(central, *band) / total for name, band in BANDS.items()}
    values.update(
        c_log_power=np.log10(total),
        o_rel_alpha=band_power(occipital, 8, 12) / (band_power(occipital, 0.5, 30) + EPS),
        eog_high=np.log10(band_power(spectra["E1-M2"], 2, 8) + EPS),
    )
    track_frame = pd.DataFrame({name: np.asarray(values[name], np.float32) for name in tracks})
    track_frame.insert(0, "second", np.arange(n))
    quality = channel_quality(signals, spectra, signals["ECG"], n, channels=FRONTAL_LEADS)
    return track_frame, quality


def extract_subject(subject: str, work: Path, *, smoke: bool) -> None:
    import run_staging_complementary_markers as markers
    from run_staging_feature_tuning import FEATURES, TRACKS

    recipe = work / "recipe"
    spec = DATASETS["cinc2018"]
    DATASETS["cinc2018"] = replace(spec, cache_dir=recipe)
    markers_cache = markers.CACHE
    markers.CACHE = work
    try:
        night = CompactNight(subject)
        columns = [name for name in FEATURES if name not in DERIVED_FEATURES]
        start = monotonic()
        stage_recipe(subject, recipe, columns, night, smoke=smoke)
        recipe_seconds = monotonic() - start
        rows = subject_rows6("cinc2018", subject, columns, work / "rows6")
        scored = subject_epochs(rows, [])[[*KEYS, TARGET]]
        if scored.empty or scored.duplicated(KEYS).any() or not scored[TARGET].isin(range(5)).all():
            raise ValueError(f"{subject}: missing, duplicate or invalid scored epochs")
        start = monotonic()
        track_names = list(dict.fromkeys([*TRACKS, "c_rel_sigma", "c_rel_alpha", "c_rel_beta"]))
        tracks, quality = extract_seconds(night, track_names)
        parquet(tracks, work / "tracks1s" / f"{subject}.parquet")
        parquet(quality.loc[:, markers.FRONTAL_QUALITY], work / "quality1s" / f"{subject}.parquet")
        per_second_seconds = monotonic() - start
        start = monotonic()
        waveforms = WaveformStore(RAW, permitted_subjects=[subject], max_open_nights=1)
        features = markers.extract(waveforms, subject)
        markers.check_night(features, scored.epoch_index.to_numpy(), subject)
        parquet(features, work / "markers.parquet", index=None)
        parquet(scored, work / "scored.parquet")
        print(
            f"{subject}: recipe {recipe_seconds:.1f}s, second-level {per_second_seconds:.1f}s, "
            f"marker panels {monotonic() - start:.1f}s, {len(scored)} scored epochs",
            flush=True,
        )
    finally:
        DATASETS["cinc2018"] = spec
        markers.CACHE = markers_cache


def prepare(subject: str, *, smoke: bool = False) -> int:
    with TemporaryDirectory(dir=SCRATCH, prefix=f"{subject}-") as directory:
        work = Path(directory)
        for family in (*FAMILIES, "quality1s"):
            (work / family).mkdir()
        extract_subject(subject, work, smoke=smoke)
        if smoke:
            verify_smoke(subject, work)
            return peak_rss_mib()
        for family in FAMILIES:
            filename = f"{subject}.parquet"
            (work / family / filename).replace(WINDOW / family / filename)
        (work / "markers.parquet").replace(MARKERS / f"{subject}.parquet")
        (work / "scored.parquet").replace(SCORED / f"{subject}.parquet")
    return peak_rss_mib()


def verify_smoke(subject: str, work: Path) -> None:
    from run_staging_feature_tuning import FEATURES, TRACKS
    from physionet_sleep.experiments.modern_tabular import _dense_cached_night

    archive = EXPLORATION / "cinc-complementary-markers/feature_cache"
    families = (*FAMILIES, "quality1s")
    expected_paths = {family: ARCHIVE_WINDOW / family / f"{subject}.parquet" for family in families}
    paths = {family: work / family / f"{subject}.parquet" for family in families}
    paths["markers"] = work / "markers.parquet"
    expected_paths["markers"] = archive / f"{subject}.parquet"
    expected = pd.read_parquet(BASE / "predictions.parquet", columns=[*KEYS, TARGET], filters=[("subject_id", "==", subject)])
    actual = pd.read_parquet(work / "scored.parquet")
    pd.testing.assert_frame_equal(actual, expected.reset_index(drop=True), check_dtype=False)
    for family, path in paths.items():
        computed = pd.read_parquet(path)
        pd.testing.assert_frame_equal(
            computed,
            pd.read_parquet(expected_paths[family], columns=computed.columns.tolist()),
            check_exact=True,
        )
    np.testing.assert_array_equal(
        _dense_cached_night(work, subject, TRACKS, FEATURES),
        _dense_cached_night(ARCHIVE_WINDOW, subject, TRACKS, FEATURES),
    )
    print(f"{subject}: model inputs, frontal quality, markers and scored keys exactly match the archive", flush=True)


def completed(subject: str) -> bool:
    filename = f"{subject}.parquet"
    return all(
        (folder / filename).is_file()
        for folder in (SCORED, MARKERS, *(WINDOW / family for family in FAMILIES))
    )


def ready(subject: str, size: int) -> bool:
    signal = RAW / subject / f"{subject}.mat"
    if not signal.is_file():
        return False
    if signal.stat().st_size != size:
        raise IOError(f"{signal}: size differs from manifest, repair acquisition before preprocessing")
    return True


@contextmanager
def cache_slot():
    # An OS-held lock is released if the coordinator or the machine exits.
    with (OUT / "cache-build.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def build(sizes: dict[str, int], workers: int, wait_minutes: float, memory_bytes: int) -> int:
    peak = {subject: estimated_peak(size) for subject, size in sizes.items()}
    # Admit the largest available reservation first and fill what space it leaves.
    # Reserving all large nights for the end often leaves only one active worker.
    remaining = {
        subject: sizes[subject]
        for subject in sorted(sizes, key=peak.__getitem__, reverse=True)
        if not completed(subject)
    }
    done = len(sizes) - len(remaining)
    if any(peak[subject] > memory_bytes for subject in remaining):
        raise ValueError("The worker memory budget cannot hold the longest remaining night")
    failed = set()
    print(
        f"Starting expansion with {done}/{len(sizes)} subjects cached, "
        f"at most {workers} workers and {memory_bytes / GIB:.2f} GiB aggregate memory",
        flush=True,
    )
    idle_until = monotonic() + wait_minutes * 60
    # FFT workspaces and allocator arenas for different night lengths can persist
    # even after the Python arrays have been released. Recycle after each night.
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        running = {}
        while remaining or running:
            if not (OUT / "PAUSE_CACHE").exists():
                available = min(
                    memory_bytes - sum(peak[subject] for subject in running.values()),
                    available_memory() - 650 * MIB,
                )
                for subject in list(remaining):
                    if len(running) >= workers:
                        break
                    if peak[subject] > available:
                        continue
                    try:
                        if not ready(subject, remaining[subject]):
                            continue
                    except OSError as error:
                        print(str(error), flush=True)
                        failed.add(subject)
                        del remaining[subject]
                        continue
                    del remaining[subject]
                    running[pool.submit(prepare, subject)] = subject
                    available -= peak[subject]
            if not running:
                if not remaining or (OUT / "PAUSE_CACHE").exists() or monotonic() >= idle_until:
                    break
                sleep(5)
                continue
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                subject = running.pop(future)
                try:
                    observed_peak = future.result()
                except Exception as error:
                    failed.add(subject)
                    print(f"{subject} failed: {error!r}", flush=True)
                else:
                    done += 1
                    print(
                        f"{done}/{len(sizes)} subjects cached ({subject}, peak RSS {observed_peak} MiB)",
                        flush=True,
                    )
            idle_until = monotonic() + wait_minutes * 60
    if remaining or failed:
        print(f"Incomplete expansion. Pending {len(remaining)}; failed {sorted(failed)}", flush=True)
        return 1
    epochs = pd.concat([pd.read_parquet(SCORED / f"{subject}.parquet") for subject in sizes], ignore_index=True)
    if set(epochs.subject_id) != sizes.keys() or epochs.duplicated(KEYS).any():
        raise ValueError("Combined expanded-cohort epochs do not agree with the manifest")
    parquet(epochs, OUT / "additional_epochs.parquet")
    print(f"Published {len(epochs)} scored epochs from {len(sizes)} training-only subjects", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1, help="simultaneous whole-night extractions")
    parser.add_argument("--memory-gib", type=float, default=4.0, help="aggregate worker memory allowance")
    parser.add_argument("--wait-minutes", type=float, default=60, help="idle time when awaiting downloads")
    parser.add_argument("--smoke", metavar="DEV_SUBJECT", help="compare a fixed development night in scratch only")
    args = parser.parse_args()
    if not 1 <= args.workers <= 3 or args.wait_minutes < 0 or not np.isfinite(args.memory_gib) or args.memory_gib <= 0:
        parser.error("choose 1–3 extraction workers, a positive memory allowance and a nonnegative wait")
    dev, holdout = development_and_holdout()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        if args.smoke not in dev:
            parser.error("smoke subject must be in the original development manifest")
        prepare(args.smoke, smoke=True)
        return 0
    sizes = expansion_manifest(dev, holdout)
    for path in (MARKERS, SCORED, *(WINDOW / family for family in FAMILIES)):
        path.mkdir(parents=True, exist_ok=True)
    with cache_slot():
        return build(sizes, args.workers, args.wait_minutes, int(args.memory_gib * GIB))


if __name__ == "__main__":
    raise SystemExit(main())
