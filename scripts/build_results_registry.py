"""Build the results registry: every number the dashboard shows, with where it came from.

Writes showcase/content/generated/results-registry.json, which the dashboard reads through showcase/lib/results.ts
(r(), fmt(), pct()). The dashboard source must not contain hand-typed results: showcase/scripts/check-results.mjs
fails the build if it does, and tests/test_results_registry.py re-checks every entry against its source.

Entry kinds
  measured    read from a saved artifact (CSV, JSON or predictions); the file's sha256 is stored with it
  derived     arithmetic on measured entries (differences, ranges); the inputs are listed in `from`
  recorded    historical results with no machine-readable artifact, kept in results/recorded.json; each must
              appear verbatim in its cited source file or in its row of docs/experiment-history.md
  literature  published figures, kept in results/recorded.json with their reference number
  protocol    rules fixed before results were seen (adoption thresholds), kept in results/recorded.json

Usage: python scripts/build_results_registry.py          (reads artifacts; ~1 min, recomputes the no-EEG ensemble)
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
WINDOW = "artifacts/staging_window_model/cinc2018"
NO_EEG = "artifacts/exploration/no-eeg"
RECORDED = ROOT / "results" / "recorded.json"
OUTPUT = ROOT / "showcase" / "content" / "generated" / "results-registry.json"
STAGES = ["Wake", "N1", "N2", "N3", "REM"]
DECODERS = {"epoch-only": "epoch", "hmm": "hmm"}

Entry = tuple[str, float, str]  # id, value, label


@dataclass(frozen=True)
class Source:
    path: str                     # primary file, relative to the repo root (hashed; its presence enables the test)
    read: Callable[[Path], list[Entry]]
    exp: int | None = None
    also: tuple[str, ...] = ()    # further files the reader uses (hashed too)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _cv_summary(label: str) -> Callable[[Path], list[Entry]]:
    """Window-model style summary: one row per config and decoder, mean of fold macro-F1."""
    def read(path: Path) -> list[Entry]:
        out = []
        for row in _rows(path):
            key = f"cv.{row['config']}.{DECODERS[row['decoder']]}"
            what = f"{label}: {row['config']} ({row['decoder']}), Boston 5-fold CV"
            out += [(f"{key}.macro_f1", float(row["macro_f1"]), f"{what}, macro-F1"),
                    (f"{key}.kappa", float(row["kappa"]), f"{what}, Cohen's kappa")]
            out += [(f"{key}.f1_{s}", float(row[f"f1_{s}"]), f"{what}, {s} F1") for s in STAGES]
        return out
    return read


def _complexity(path: Path) -> list[Entry]:
    out = []
    for row in _rows(path):
        key = f"complexity.{row['learner']}"
        out += [(f"{key}.macro_f1", float(row["macro_f1"]), f"Model C inputs, learner {row['learner']}, macro-F1"),
                (f"{key}.learned_values", float(row["learned_values"]), f"learner {row['learner']}, learned values (fold mean)")]
    return out


def _holdout(look: int) -> Callable[[Path], list[Entry]]:
    def read(path: Path) -> list[Entry]:
        out = []
        for row in _rows(path):
            key = f"holdout.look{look}.{row['config']}"
            what = f"Held-out Boston (200 patients), look #{look}, {row['config']}"
            out += [(f"{key}.{m}", float(row[m]), f"{what}, {m}") for m in ("macro_f1", "kappa", "accuracy")]
        return out
    return read


def _ledger(path: Path) -> list[Entry]:
    look1 = [float(r["adopted_macro_f1"]) for r in _rows(path) if r["look"] == "1"]
    return [("holdout.look1.rerun_shift", abs(look1[-1] - look1[0]),
             "Held-out look #1: change after rebuilding the one failed record and rerunning the same batch")]


def _coimbra(path: Path) -> list[Entry]:
    out = []
    for row in _rows(path):
        key = f"coimbra.{row['evaluation']}"
        out += [(f"{key}.macro_f1", float(row["macro_f1"]), f"Coimbra, {row['evaluation']}, macro-F1"),
                (f"{key}.kappa", float(row["kappa"]), f"Coimbra, {row['evaluation']}, kappa")]
    return out


def _ucddb_transfer(path: Path) -> list[Entry]:
    out = []
    for row in _rows(path):
        key = f"ucddb_transfer.d27.{row['evaluation']}"
        out += [(f"{key}.macro_f1", float(row["macro_f1"]), f"27-signal shared-lead model, {row['evaluation']}, macro-F1"),
                (f"{key}.kappa", float(row["kappa"]), f"27-signal shared-lead model, {row['evaluation']}, kappa")]
    return out


def _no_eeg_sequence(path: Path) -> list[Entry]:
    out = []
    for row in _rows(path):
        key = f"noeeg.{row['variant']}.{DECODERS[row['decoder']]}"
        what = f"No-EEG whole-night network {row['variant']} ({row['decoder']}, seeds {row['seeds']})"
        out += [(f"{key}.macro_f1", float(row["macro_f1"]), f"{what}, macro-F1"),
                (f"{key}.kappa4", float(row["kappa4"]), f"{what}, 4-class kappa"),
                (f"{key}.f1_N1", float(row["f1_N1"]), f"{what}, N1 F1")]
    return out


def _no_eeg_ensemble(path: Path) -> list[Entry]:
    """Post hoc mean probability of 4 GRU and 4 TCN seeds, scored on the window run's folds (exp #31)."""
    import numpy as np
    from sklearn.metrics import cohen_kappa_score, f1_score
    sys.path.insert(0, str(ROOT / "scripts"))
    import export_staging_dashboard_data as ex
    preds, splits = ex._prediction_frame()
    y = preds["y_true"].to_numpy()
    gru = np.load(path)["proba"]
    tcn = np.load(ROOT / NO_EEG / "sequence" / "proba_tcn_base_fine_raw_pos_4seeds.npz")["proba"]
    if gru.shape != (len(y), 5) or tcn.shape != gru.shape:
        raise SystemExit("no-EEG probabilities do not match the window run's epochs")
    pred = ((gru + tcn) / 2).argmax(1)
    four = np.array([0, 1, 1, 2, 3])
    folds = [np.asarray(s.test_idx) for s in splits]
    what = "No-EEG post hoc ensemble (4 GRU + 4 TCN seeds), heart, breathing and oximetry, whole night"
    return [("noeeg.ensemble.macro_f1", float(np.mean([f1_score(y[f], pred[f], average="macro") for f in folds])), f"{what}, macro-F1"),
            ("noeeg.ensemble.kappa4", float(np.mean([cohen_kappa_score(four[y[f]], four[pred[f]]) for f in folds])), f"{what}, 4-class kappa"),
            ("noeeg.ensemble.f1_N1", float(f1_score(y, pred, labels=range(5), average=None)[1]), f"{what}, N1 F1")]


def _signal_ablations(path: Path) -> list[Entry]:
    return [(f"cv.{row['config']}.{DECODERS[row['decoder']]}.kappa4", float(row["kappa4"]),
             f"{row['config']} ({row['decoder']}), 4-class kappa (Wake / N1+N2 / N3 / REM)")
            for row in json.loads(path.read_text(encoding="utf-8"))["signal_ablations"]]


def _effect_sizes(path: Path) -> list[Entry]:
    return [(f"effect.{r['measure']}.d_N2", float(r["d_N2_raw"]), f"{r['measure']}: N2 vs rest, Cohen's d (raw)")
            for r in _rows(path)]


MEASURED = [
    Source(f"{WINDOW}/summary.csv", _cv_summary("Window model"), exp=23),
    Source(f"{WINDOW}/ablations/summary.csv", _cv_summary("Signal ablation"), exp=24),
    Source(f"{WINDOW}/cardio_full/summary.csv", _cv_summary("Heart and breathing only"), exp=26),
    Source(f"{WINDOW}/feature_reduction/summary.csv", _cv_summary("Feature reduction"), exp=33),
    Source(f"{WINDOW}/complexity/summary.csv", _complexity, exp=25),
    Source(f"{WINDOW}/holdout/summary_look1.csv", _holdout(1), exp=30),
    Source(f"{WINDOW}/holdout/summary.csv", _holdout(2), exp=37),
    Source(f"{WINDOW}/holdout/LEDGER.csv", _ledger, exp=30),
    Source("artifacts/staging_window_model/transfer_isruc/summary.csv", _coimbra, exp=38),
    Source("artifacts/staging_window_model/transfer/summary_d27.csv", _ucddb_transfer, exp=27),
    Source(f"{NO_EEG}/sequence/summary.csv", _no_eeg_sequence, exp=31),
    Source(f"{NO_EEG}/sequence/proba_gru_base_fine_raw_pos_4seeds.npz", _no_eeg_ensemble, exp=31,
           also=(f"{NO_EEG}/sequence/proba_tcn_base_fine_raw_pos_4seeds.npz", f"{WINDOW}/predictions.parquet")),
    Source("showcase/content/generated/staging-dashboard.json", _signal_ablations, exp=26),
    Source(f"{WINDOW}/feature_reduction/effect_sizes.csv", _effect_sizes, exp=32),
]


def derive(v: dict[str, float]) -> dict[str, tuple[float, list[str], str]]:
    """Derived entries: id -> (value, input ids, label). Pure arithmetic on measured values."""
    def diff(a: str, b: str, label: str):
        return v[a] - v[b], [a, b], label
    spindle = ["cv.nocardio27_bursts.epoch.macro_f1", "cv.nocardio27.epoch.macro_f1",
               "cv.prune6_sigratio.epoch.macro_f1", "cv.prune6.epoch.macro_f1", "cv.prune6_bursts.epoch.macro_f1"]
    cardio = [f"cv.{c}.{d}.macro_f1" for c in ("cardio6", "cardio6_c5") for d in ("epoch", "hmm")]
    return {
        "delta.cardio_with_eeg": diff("cv.window33.epoch.macro_f1", "cv.nocardio27.epoch.macro_f1",
                                      "Gain from adding heart and breathing to EEG, eyes and chin (B minus 27 signals)"),
        "delta.decoder_with_window": diff("cv.window33.hmm.macro_f1", "cv.window33.epoch.macro_f1",
                                          "Gain from the whole-night decoder on the 33-signal window model"),
        "delta.spindle_additions_best": (max(v[spindle[0]] - v[spindle[1]], v[spindle[2]] - v[spindle[3]], v[spindle[4]] - v[spindle[3]]),
                                         spindle, "Largest gain from adding a spindle measure (exp #33)"),
        "delta.swap_spindle": diff("cv.swapS2.epoch.macro_f1", "cv.prune6.epoch.macro_f1",
                                   "Spindle-burst swap (model C) minus the 21-signal model"),
        "range.cardio_alone.low": (min(v[c] for c in cardio), cardio, "Heart and breathing alone, 6 signals: lowest of 2.5/5.5 min, with or without decoder"),
        "range.cardio_alone.high": (max(v[c] for c in cardio), cardio, "Heart and breathing alone, 6 signals: highest of 2.5/5.5 min, with or without decoder"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source(source: Source) -> dict[str, float]:
    return {entry_id: value for entry_id, value, _ in source.read(ROOT / source.path)}


def audit_doc() -> int:
    """List numbers in docs/experiment-history.md rows that no registry entry (at any precision) explains."""
    import re
    entries = json.loads(OUTPUT.read_text(encoding="utf-8"))["entries"]
    known = {f"{abs(e['value']):.{d}f}" for e in entries.values() for d in (2, 3, 4)}
    unexplained = 0
    for line in (ROOT / "docs" / "experiment-history.md").read_text(encoding="utf-8").splitlines():
        row = re.match(r"\|\s*(\d+)\s*\|", line)
        if not row:
            continue
        missing = sorted({n for n in re.findall(r"(?<![\w.])\d\.\d{2,4}(?![\d])", line) if n not in known})
        if missing:
            unexplained += len(missing)
            print(f"#{row.group(1)}: {' '.join(missing)}")
    print(f"{unexplained} numbers in experiment-history.md have no registry entry")
    return 0


def main() -> int:
    if "--audit-doc" in sys.argv:
        return audit_doc()
    entries: dict[str, dict] = {}
    files: dict[str, str] = {}
    for source in MEASURED:
        path = ROOT / source.path
        if not path.exists():
            raise SystemExit(f"missing source {source.path}: the registry is built from artifacts, not copied values")
        for name in (source.path, *source.also):
            files[name] = sha256(ROOT / name)
        for entry_id, value, label in source.read(path):
            if entry_id in entries and abs(entries[entry_id]["value"] - value) > 1e-12:
                raise SystemExit(f"{entry_id}: {source.path} disagrees with {entries[entry_id]['source']}")
            entries.setdefault(entry_id, {"value": value, "kind": "measured", "label": label, "source": source.path,
                                          "exp": source.exp})
    values = {k: e["value"] for k, e in entries.items()}
    for entry_id, (value, inputs, label) in derive(values).items():
        entries[entry_id] = {"value": value, "kind": "derived", "label": label, "from": inputs}
    for item in json.loads(RECORDED.read_text(encoding="utf-8")):
        if item["id"] in entries:
            raise SystemExit(f"{item['id']} is recorded by hand but also measured; keep the measured one")
        if item["kind"] not in ("recorded", "literature", "protocol"):
            raise SystemExit(f"{item['id']}: unknown kind {item['kind']}")
        if item["kind"] == "literature" and "cite" not in item:
            raise SystemExit(f"{item['id']}: literature entries need a reference number")
        entries[item.pop("id")] = item
    payload = {"generated_by": "scripts/build_results_registry.py", "files": files,
               "entries": dict(sorted(entries.items()))}
    OUTPUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    kinds = {}
    for e in entries.values():
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(entries)} entries {kinds} from {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
