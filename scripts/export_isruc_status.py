"""Export a lightweight external-cohort status without rerunning any analysis."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raw = ROOT / "data/raw/isruc"
    ids = range(1, 101)
    ready = [s for s in ids if all((raw / str(s) / f"{s}{suffix}").exists()
                                  for suffix in (".rec", "_1.txt", "_2.txt"))]
    processed = [s for s in ready if
                 (ROOT / f"artifacts/isruc_recipe_cache/{s}/final__feature_table_py").exists()
                 and (ROOT / f"artifacts/isruc_window_cache/tracks1s/{s}.parquet").exists()]
    folder = ROOT / "artifacts/staging_window_model/transfer_isruc"
    summary = folder / "summary.csv"
    rows = list(csv.DictReader(summary.open(newline=""))) if summary.exists() else []
    look_path = folder / "single_look.json"
    look = json.loads(look_path.read_text()) if look_path.exists() else {}
    included = {int(s) for s in look.get("isruc_nights", [])}
    excluded = {int(s): reason for s, reason in look.get("excluded", {}).items()}
    subject_path = folder / "isruc_subjects.csv"
    subjects = list(csv.DictReader(subject_path.open(newline=""))) if subject_path.exists() else []
    # Require a recorded evaluation covering every eligible night and explicit exclusions.
    complete = (bool(rows) and look.get("config") == "g20" and bool(included)
                and not included & excluded.keys() and included | excluded.keys() == set(ids)
                and included <= set(processed)
                and {int(row["subject_id"]) for row in subjects} == included
                and (folder / "predictions.parquet").exists())
    if complete:
        complete = all(int(row["n_subjects"]) == len(included) and row.get("config") == "g20"
                       for row in rows if row["evaluation"] != "in_lab_cinc_cv")
    out = {
        "updated": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="minutes"),
        "downloaded": len(ready), "processed": len(processed),
        "missing": [s for s in ids if s not in ready],
        "pending_processing": [s for s in ready if s not in processed],
        "final_complete": complete,
        "evaluated": len(included) if complete else 0,
        "excluded": {str(s): reason for s, reason in excluded.items()},
        "results": [{"evaluation": row["evaluation"], "macro_f1": float(row["macro_f1"]),
                     "accuracy": float(row["accuracy"]), "kappa": float(row["kappa"]),
                     "n_subjects": int(row["n_subjects"]),
                     "stage_f1": {stage: float(row[f"f1_{stage}"]) for stage in ("Wake", "N1", "N2", "N3", "REM")}}
                    for row in rows] if complete else [],
        "subject_macro_f1_scorer1": [float(row["macro_f1_scorer1"]) for row in
                                     csv.DictReader((folder / "isruc_subjects.csv").open(newline=""))]
                                    if complete and (folder / "isruc_subjects.csv").exists() else [],
    }
    path = ROOT / "showcase/content/generated/isruc-status.json"
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
