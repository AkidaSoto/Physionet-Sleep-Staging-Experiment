"""Heart, breathing and SpO2 inputs from raw CinC 2018 records for the no-EEG staging experiments.

Per subject (development subjects only), written to artifacts/exploration/no-eeg/cache/:
  epochs/<id>.parquet   one row per 30-s epoch (epoch i covers [30 i, 30 i + 30) s), raw-scale summaries:
      heart:    hr_mean, hr_sd, hr_min, hr_max, ln_rmssd, ln_sdnn, pnn50, ln_lf, ln_hf, lf_hf (120-s window)
      breath:   br_rate, br_interval_cv, br_amp_log, br_amp_cv, br_count, resp_peak_hz (120-s window),
                abd_chest_corr, effort_env_log
      movement: effort_move (baseline jumps of the belts), effort_hf (0.8-5 Hz belt power), ecg_noise (70-95 Hz ECG power)
      SpO2:     spo2_mean, spo2_min, spo2_sd, spo2_drop (mean - min)
  ihr4/<id>.npy          instantaneous heart rate at 4 Hz (bpm, NaN where no clean beats)
  sec/<id>.parquet       1-s tracks: ihr, br_rate, br_amp_log, effort_move, effort_hf, ecg_noise, spo2

Usage: python scripts/build_noeeg_cache.py [--jobs 2]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data/raw/cinc2018"
PEAKS = REPO / "artifacts/cinc2018_recipe_cache"
OUT = REPO / "artifacts/exploration/no-eeg/cache"
FS = 200
EPOCH = 30
EPS = 1e-6


def clean_rr(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(beat-pair midpoints, RR) keeping 0.3-2.0 s intervals within 30 % of the local median."""
    rr = np.diff(r)
    mid = (r[1:] + r[:-1]) / 2
    ok = (rr > 0.3) & (rr < 2.0)
    if ok.sum() < 20:
        return mid[:0], rr[:0]
    med = pd.Series(np.where(ok, rr, np.nan)).rolling(11, center=True, min_periods=3).median().to_numpy()
    ok &= np.abs(rr - med) < 0.3 * med
    return mid[ok], rr[ok]


def interp_grid(t: np.ndarray, v: np.ndarray, grid: np.ndarray, max_gap: float) -> np.ndarray:
    """Linear interpolation of (t, v) onto grid, NaN where the nearest sample is more than max_gap away."""
    if len(t) < 2:
        return np.full(len(grid), np.nan)
    out = np.interp(grid, t, v, left=np.nan, right=np.nan)
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    gap = np.minimum(np.abs(grid - t[j - 1]), np.abs(t[j] - grid))
    out[gap > max_gap] = np.nan
    return out


def band_log_power(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    if np.isnan(x).mean() > 0.2:
        return np.nan
    x = np.nan_to_num(x - np.nanmean(x))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 / len(x)
    return float(np.log(p[(f >= lo) & (f < hi)].sum() + EPS))


def per_second(x: np.ndarray, n: int, fs: int, fn) -> np.ndarray:
    return fn(x[: n * fs].reshape(n, fs), axis=1)


def subject_cache(sid: str) -> None:
    import wfdb
    from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

    rec = wfdb.rdrecord(str(RAW / sid / sid), channel_names=["ABD", "CHEST", "SaO2", "ECG"], physical=True)
    sig = {name: np.nan_to_num(rec.p_signal[:, i]) for i, name in enumerate(rec.sig_name)}
    n = len(sig["ABD"]) // FS
    ne = n // EPOCH
    sec_grid = np.arange(n) + 0.5

    # ---- heart
    r = pd.read_parquet(PEAKS / sid / "pan_tompkins_py/events/r_peaks.parquet")["start"].to_numpy() / FS
    mid, rr = clean_rr(r)
    g4 = np.arange(n * 4) / 4 + 0.125
    ihr4 = interp_grid(mid, 60 / rr, g4, 3.0).astype(np.float32)
    ihr1 = np.nanmean(ihr4.reshape(n, 4), 1) if n else ihr4
    ecg_hp = sosfiltfilt(butter(4, [70, 95], btype="band", fs=FS, output="sos"), sig["ECG"])
    ecg_noise = 0.5 * np.log10(per_second(ecg_hp ** 2, n, FS, np.mean) + 1e-12)

    # ---- breathing (summed belts at 10 Hz)
    def to10(x):
        return x[: n * FS].reshape(n * 10, FS // 10).mean(1)

    eff = to10(sig["ABD"] + sig["CHEST"])
    bp = sosfiltfilt(butter(4, [0.1, 0.7], btype="band", fs=10, output="sos"), eff)
    env = np.abs(hilbert(bp))
    env_med = pd.Series(env).rolling(600, center=True, min_periods=50).median().to_numpy()
    pk, _ = find_peaks(bp, distance=15, prominence=0.4 * np.nan_to_num(env_med, nan=np.nanmedian(env)))
    tr, _ = find_peaks(-bp, distance=15)
    bt = pk / 10.0
    amp = np.full(len(pk), np.nan)
    if len(tr) > 1 and len(pk) > 1:
        j = np.clip(np.searchsorted(tr, pk), 1, len(tr) - 1)
        amp = bp[pk] - (bp[tr[j - 1]] + bp[tr[j]]) / 2
    bint = np.diff(bt)
    bmid = (bt[1:] + bt[:-1]) / 2
    okb = (bint > 1.5) & (bint < 15)
    br_rate1 = interp_grid(bmid[okb], 60 / bint[okb], sec_grid, 15.0)
    amp_ok = np.isfinite(amp) & (amp > 0)
    br_amp1 = interp_grid(bt[amp_ok], np.log10(amp[amp_ok] + EPS), sec_grid, 15.0)
    base = sosfiltfilt(butter(2, 0.05, btype="low", fs=10, output="sos"), eff)
    base1 = base.reshape(n, 10).mean(1)
    effort_move = np.log10(np.abs(np.diff(base1, prepend=base1[:1])) + 1e-3)
    raw200 = (sig["ABD"] + sig["CHEST"])[: n * FS]
    hf = sosfiltfilt(butter(4, [0.8, 5], btype="band", fs=FS, output="sos"), raw200)
    effort_hf = 0.5 * np.log10(per_second(hf ** 2, n, FS, np.mean) + 1e-6)
    abd10 = sosfiltfilt(butter(4, [0.1, 0.7], btype="band", fs=10, output="sos"), to10(sig["ABD"]))
    chest10 = sosfiltfilt(butter(4, [0.1, 0.7], btype="band", fs=10, output="sos"), to10(sig["CHEST"]))

    # ---- SpO2
    sp = per_second(sig["SaO2"], n, FS, np.mean)
    sp[(sp < 50) | (sp > 100.5)] = np.nan

    rows = []
    for e in range(ne):
        a, b = e * EPOCH, (e + 1) * EPOCH
        h = ihr4[a * 4: b * 4]
        m = (mid >= a) & (mid < b)
        rre = rr[m]
        row = {"epoch_index": e}
        row["hr_mean"] = np.nanmean(h) if np.isfinite(h).sum() > 20 else np.nan
        row["hr_sd"] = np.nanstd(h) if np.isfinite(h).sum() > 20 else np.nan
        row["hr_min"] = np.nanmin(h) if np.isfinite(h).sum() > 20 else np.nan
        row["hr_max"] = np.nanmax(h) if np.isfinite(h).sum() > 20 else np.nan
        if len(rre) > 10:
            d = np.diff(rre)
            row["ln_rmssd"] = np.log(np.sqrt(np.mean(d ** 2)) * 1000 + 1)
            row["ln_sdnn"] = np.log(np.std(rre) * 1000 + 1)
            row["pnn50"] = float(np.mean(np.abs(d) > 0.05))
        c = (a + b) / 2
        w = ihr4[max(0, int((c - 60) * 4)): int((c + 60) * 4)]
        if len(w) == 480:
            lf, hfp = band_log_power(w, 4, 0.04, 0.15), band_log_power(w, 4, 0.15, 0.4)
            row.update(ln_lf=lf, ln_hf=hfp, lf_hf=lf - hfp)
        bm = (bmid >= a) & (bmid < b) & okb
        row["br_count"] = int(((bt >= a) & (bt < b)).sum())
        if bm.sum() >= 3:
            bi = bint[bm]
            row["br_rate"] = 60 / bi.mean()
            row["br_interval_cv"] = bi.std() / bi.mean()
        am = amp[(bt >= a) & (bt < b) & amp_ok]
        if len(am) >= 3:
            row["br_amp_log"] = float(np.log10(np.median(am) + EPS))
            row["br_amp_cv"] = float(am.std() / am.mean())
        ws = bp[max(0, int((c - 60) * 10)): int((c + 60) * 10)]
        if len(ws) == 1200:
            f = np.fft.rfftfreq(1200, 0.1)
            p = np.abs(np.fft.rfft(ws * np.hanning(1200))) ** 2
            band = (f >= 0.1) & (f < 0.7)
            row["resp_peak_hz"] = float(f[band][np.argmax(p[band])])
        x1, x2 = abd10[a * 10: b * 10], chest10[a * 10: b * 10]
        if x1.std() > 0 and x2.std() > 0:
            row["abd_chest_corr"] = float(np.corrcoef(x1, x2)[0, 1])
        row["effort_env_log"] = float(np.log10(env[a * 10: b * 10].mean() + EPS))
        row["effort_move"] = float(effort_move[a:b].max())
        row["effort_hf"] = float(np.mean(effort_hf[a:b]))
        row["ecg_noise"] = float(np.percentile(ecg_noise[a:b], 90))
        s = sp[a:b]
        if np.isfinite(s).sum() > 15:
            row.update(spo2_mean=np.nanmean(s), spo2_min=np.nanmin(s), spo2_sd=np.nanstd(s),
                       spo2_drop=np.nanmean(s) - np.nanmin(s))
        rows.append(row)
    cols = ["epoch_index", "hr_mean", "hr_sd", "hr_min", "hr_max", "ln_rmssd", "ln_sdnn", "pnn50", "ln_lf", "ln_hf",
            "lf_hf", "br_rate", "br_interval_cv", "br_amp_log", "br_amp_cv", "br_count", "resp_peak_hz",
            "abd_chest_corr", "effort_env_log", "effort_move", "effort_hf", "ecg_noise", "spo2_mean", "spo2_min",
            "spo2_sd", "spo2_drop"]
    df = pd.DataFrame(rows).reindex(columns=cols).astype(np.float32)
    df["epoch_index"] = df["epoch_index"].astype(int)
    for sub in ("epochs", "ihr4", "sec"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "epochs" / f"{sid}.parquet", index=False)
    np.save(OUT / "ihr4" / f"{sid}.npy", ihr4)
    pd.DataFrame({"ihr": ihr1, "br_rate": br_rate1, "br_amp_log": br_amp1, "effort_move": effort_move,
                  "effort_hf": effort_hf, "ecg_noise": ecg_noise, "spo2": sp}).astype(np.float32).to_parquet(
        OUT / "sec" / f"{sid}.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--subjects", default="")
    args = ap.parse_args()
    dev = (REPO / "artifacts/staging_window_model/cinc2018/dev_subjects.txt").read_text().split()
    holdout = set((RAW / "selected_records_holdout.txt").read_text().split()) if (RAW / "selected_records_holdout.txt").exists() else set()
    subjects = [s for s in (args.subjects.split(",") if args.subjects else dev) if s in dev and s not in holdout]
    todo = [s for s in subjects if not (OUT / "epochs" / f"{s}.parquet").exists()]
    print(f"{len(todo)} of {len(subjects)} subjects to build", flush=True)
    t0 = time.time()
    if args.jobs <= 1:
        for i, s in enumerate(todo):
            subject_cache(s)
            print(f"{i + 1}/{len(todo)} {s} {time.time() - t0:.0f}s", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.jobs) as ex:
            for i, (s, _) in enumerate(zip(todo, ex.map(subject_cache, todo))):
                print(f"{i + 1}/{len(todo)} {s} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
