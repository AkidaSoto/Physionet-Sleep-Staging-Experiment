from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--backends", nargs="+", default=("adaptive", "wfdb_xqrs", "sleepecg"))
    parser.add_argument("--output-dir", default="artifacts/ecg_backend_comparison")
    return parser.parse_args()


def _rr_summary(r_peak_idx: np.ndarray, fs: float) -> dict[str, float]:
    rr = np.diff(np.asarray(r_peak_idx, dtype=float)) / float(fs) if len(r_peak_idx) >= 2 else np.array([], dtype=float)
    hr = 60.0 / rr if rr.size else np.array([], dtype=float)
    return {
        "n_rr": int(rr.size),
        "rr_lt_0p3_frac": float(np.mean(rr < 0.3)) if rr.size else 0.0,
        "rr_lt_0p25_frac": float(np.mean(rr < 0.25)) if rr.size else 0.0,
        "rr_gt_2p0_frac": float(np.mean(rr > 2.0)) if rr.size else 0.0,
        "hr_gt_180_frac": float(np.mean(hr > 180.0)) if hr.size else 0.0,
        "hr_gt_220_frac": float(np.mean(hr > 220.0)) if hr.size else 0.0,
        "rr_min_seconds": float(np.min(rr)) if rr.size else np.nan,
        "rr_max_seconds": float(np.max(rr)) if rr.size else np.nan,
        "hr_max_bpm": float(np.max(hr)) if hr.size else np.nan,
    }


def _feature_summary(feature_table: pd.DataFrame) -> dict[str, float]:
    cols = [
        "pan_tompkins.heart_rate_bpm_track",
        "pan_tompkins.hrv_rmssd_track",
        "pan_tompkins.hrv_sdnn_track",
        "pan_tompkins.heart_rate_min_track",
        "pan_tompkins.heart_rate_range_track",
    ]
    out: dict[str, float] = {}
    for col in cols:
        if col not in feature_table.columns:
            continue
        series = pd.to_numeric(feature_table[col], errors="coerce")
        short = col.replace("pan_tompkins.", "")
        out[f"{short}_q50"] = float(series.quantile(0.50))
        out[f"{short}_q95"] = float(series.quantile(0.95))
        out[f"{short}_q99"] = float(series.quantile(0.99))
        out[f"{short}_max"] = float(series.max())
    if "pan_tompkins.heart_rate_range_track" in feature_table.columns:
        hr_range = pd.to_numeric(feature_table["pan_tompkins.heart_rate_range_track"], errors="coerce")
        out["hr_range_gt_100_frac"] = float((hr_range > 100.0).mean())
        out["hr_range_gt_60_frac"] = float((hr_range > 60.0).mean())
    return out


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from physionet_sleep.algorithms import FeatureTableCompiler, PanTompkinsDetector
    from physionet_sleep.core.pipeline import AlgorithmPipeline
    from physionet_sleep.io.ucddb import discover_ucddb_records, load_ucddb_waveforms, resolve_ucddb_record

    root_dir = repo_root / "data" / "raw" / "ucddb"
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.subjects:
        subjects = list(args.subjects)
    else:
        subjects = [record.record_id for record in discover_ucddb_records(root_dir)]

    rows: list[dict[str, object]] = []
    for subject_id in subjects:
        record = load_ucddb_waveforms(resolve_ucddb_record(root_dir, subject_id))
        for backend in args.backends:
            row: dict[str, object] = {"subject_id": subject_id, "backend": str(backend)}
            try:
                pipeline = AlgorithmPipeline(
                    algorithms=[
                        PanTompkinsDetector(signal_key="ecg", backend=str(backend)),
                        FeatureTableCompiler(),
                    ]
                )
                results = pipeline.run(record)
                detector = results["pan_tompkins"]
                feature_table = results["feature_table"].artifacts.get("feature_table", pd.DataFrame())
                duration_seconds = max((series.duration_seconds for series in record.channels.values()), default=0.0)
                row["status"] = "ok"
                row["num_r_peaks"] = int(detector.features.get("num_r_peaks", 0))
                row["duration_hours"] = float(duration_seconds / 3600.0)
                row |= _rr_summary(
                    np.asarray(detector.features.get("r_peak_indices", []), dtype=int),
                    fs=float(detector.features.get("sample_rate_hz", 0.0) or 0.0),
                )
                if isinstance(feature_table, pd.DataFrame) and not feature_table.empty:
                    row |= _feature_summary(feature_table)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)

    detail = pd.DataFrame(rows)
    detail.to_parquet(output_dir / "detail.parquet", index=False)

    summary_rows: list[dict[str, object]] = []
    if not detail.empty:
        metric_cols = [
            col
            for col in detail.columns
            if col not in {"subject_id", "backend", "status", "error"} and pd.api.types.is_numeric_dtype(detail[col])
        ]
        for backend, group in detail.groupby("backend", sort=False):
            row = {
                "backend": backend,
                "n_subjects": int(group["subject_id"].nunique()),
                "n_ok": int((group["status"] == "ok").sum()),
                "n_error": int((group["status"] == "error").sum()),
            }
            ok_group = group[group["status"] == "ok"]
            for col in metric_cols:
                if col in ok_group.columns and not ok_group.empty:
                    row[f"{col}_mean"] = float(ok_group[col].mean())
                    row[f"{col}_median"] = float(ok_group[col].median())
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        sort_cols = [col for col in ("rr_lt_0p25_frac_mean", "hr_gt_220_frac_mean", "hr_range_gt_100_frac_mean") if col in summary.columns]
        if sort_cols:
            summary = summary.sort_values(
                by=sort_cols,
                ascending=True,
                na_position="last",
            )
    summary.to_parquet(output_dir / "summary.parquet", index=False)
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "subjects": subjects,
                "backends": list(args.backends),
                "detail_rows": int(len(detail)),
                "summary_rows": int(len(summary)),
            },
            handle,
            indent=2,
        )

    if not summary.empty:
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
