"""Export the sleep-staging dashboard data (CinC 2018, window model).

Reads the outputs of scripts/run_window_hmm_staging.py (predictions, permutation importance, transition matrix and
bout lengths), the attention-CNN test probabilities from artifacts/exploration/cinc-context, the 6-s feature tables
and the 1-s track cache, and writes:
  showcase/content/generated/staging-dashboard.json           models, diagnostics, importance, sleep structure
  showcase/content/generated/staging-feature-examples.json    raw-signal example epochs for the feature explorer
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
GENERATED = ROOT / "showcase" / "content" / "generated"
OUTPUT = GENERATED / "staging-dashboard.json"
FEATURE_EXAMPLES_OUTPUT = GENERATED / "staging-feature-examples.json"
STAGES = ["Wake", "N1", "N2", "N3", "REM"]
DATASET_NAME = "PhysioNet/CinC Challenge 2018 (MGH sleep laboratory)"
RECIPE_CACHE = ARTIFACTS / "cinc2018_recipe_cache"
TRACK_CACHE = ARTIFACTS / "cinc2018_window_cache" / "tracks1s"
WINDOW_RUN = ARTIFACTS / "staging_window_model" / "cinc2018"
CNN_PROBS = ARTIFACTS / "exploration" / "cinc-context" / "cache"
SUBJECT_INFO = [ARTIFACTS / "exploration" / "cinc2018" / "selected_subjects.csv",
                ARTIFACTS / "exploration" / "cinc2018" / "selected_subjects_batch2.csv"]
ECG_FS = 200.0

sys.path.insert(0, str(ROOT / "src"))
from physionet_sleep.analysis.epoch_windows import SPINDLE_BURSTS, WINDOW_FEATURES, spindle_burst_counts  # noqa: E402
from physionet_sleep.decoders.hmm import learn_transition_matrix, posterior_decode  # noqa: E402
from physionet_sleep.experiments.tabular_runner import build_subject_eval_splits  # noqa: E402

# Families group each 6-s feature with the 1-s tracks that measure the same physiology. Track ids are "track.<name>".
FEATURE_GROUPS = [
    {"id": "spectrum", "label": "EEG spectrum", "signal": "EEG", "feature_ids": [
        "spindle_features.aperiodic_exponent", "spindle_features.aperiodic_intercept", "spindle_features.delta_power",
        "epoch_ratio_features.rel_theta", "epoch_ratio_features.rel_beta", "spindle_features.theta_residual_power",
        "spindle_features.alpha_residual_power", "spindle_features.beta_residual_power",
        "spindle_features.gamma_residual_power", "track.c_rel_delta", "track.c_rel_theta", "track.c_rel_alpha",
        "track.c_rel_beta", "track.c_log_power"]},
    {"id": "spindles", "label": "Spindles", "signal": "EEG", "feature_ids": ["track.c_rel_sigma", SPINDLE_BURSTS]},
    {"id": "topography", "label": "Frontal slow waves / occipital alpha", "signal": "EEG", "feature_ids": [
        "topography_features.frontal_slow_wave_fraction", "topography_features.occipital_rel_alpha",
        "track.f_rel_delta", "track.o_rel_alpha"]},
    {"id": "eye-movement", "label": "Eye movement", "signal": "EOG", "feature_ids": [
        "eye_movement_activity.eye_movement_activity", "eye_movement_activity.eye_movement_baseline",
        "epoch_ratio_features.eog_rem_vs_noise_db", "epoch_ratio_features.eog_rem_vs_sem_db",
        "track.eog_low", "track.eog_high"]},
    {"id": "muscle-tone", "label": "Muscle tone", "signal": "Chin EMG", "feature_ids": [
        "epoch_ratio_features.emg_floor_db", "track.emg_log_rms"]},
    {"id": "heart", "label": "Heart", "signal": "ECG", "feature_ids": ["hrv_features.hr_slow_sd_300s", "track.hr"]},
    {"id": "breathing", "label": "Breathing", "signal": "Respiratory effort", "feature_ids": [
        "breath_regularity.breath_interval_cv_60s", "cardiorespiratory_coupling.hr_resp_coherence_120s",
        "track.resp_effort", "track.resp_env"]},
]

FEATURE_LABELS = {
    "spindle_features.aperiodic_exponent": "1/f slope",
    "spindle_features.aperiodic_intercept": "1/f offset",
    "spindle_features.delta_power": "Delta power",
    "epoch_ratio_features.rel_theta": "Theta share",
    "epoch_ratio_features.rel_beta": "Beta share",
    "spindle_features.theta_residual_power": "Theta residual power",
    "spindle_features.alpha_residual_power": "Alpha residual power",
    "spindle_features.beta_residual_power": "Beta residual power",
    "spindle_features.gamma_residual_power": "Gamma residual power",
    "eye_movement_activity.eye_movement_activity": "Eye-movement activity",
    "eye_movement_activity.eye_movement_baseline": "Eye-movement baseline",
    "epoch_ratio_features.eog_rem_vs_noise_db": "Eye-movement power",
    "epoch_ratio_features.eog_rem_vs_sem_db": "Fast-to-slow eye movement ratio",
    "epoch_ratio_features.emg_floor_db": "Chin muscle tone",
    "hrv_features.hr_slow_sd_300s": "Slow heart-rate swings",
    "breath_regularity.breath_interval_cv_60s": "Breath timing variability",
    "cardiorespiratory_coupling.hr_resp_coherence_120s": "Heart–breathing coupling",
    "topography_features.frontal_slow_wave_fraction": "Frontal slow-wave fraction",
    "topography_features.occipital_rel_alpha": "Occipital alpha share",
    "track.c_rel_delta": "Central delta share (1 s)",
    "track.c_rel_theta": "Central theta share (1 s)",
    "track.c_rel_alpha": "Central alpha share (1 s)",
    "track.c_rel_sigma": "Spindle-band share (1 s)",
    "track.c_rel_beta": "Central beta share (1 s)",
    "track.c_log_power": "Central EEG power (1 s)",
    "track.f_rel_delta": "Frontal delta share (1 s)",
    "track.o_rel_alpha": "Occipital alpha share (1 s)",
    "track.eog_low": "Slow EOG power (1 s)",
    "track.eog_high": "Fast EOG power (1 s)",
    "track.emg_log_rms": "Chin EMG level (1 s)",
    "track.hr": "Heart rate (1 s)",
    "track.resp_effort": "Breathing effort (1 s)",
    "track.resp_env": "Breathing amplitude (1 s)",
    SPINDLE_BURSTS: "Spindle bursts",
}

FEATURE_IDS = [feature_id for group in FEATURE_GROUPS for feature_id in group["feature_ids"]]
DERIVED_IDS = [SPINDLE_BURSTS]  # epoch features computed from the 1-s tracks (model C)
TABLE_FEATURES = [f for f in FEATURE_IDS if not f.startswith("track.") and f not in DERIVED_IDS]
TRACK_FEATURES = [f for f in FEATURE_IDS if f.startswith("track.")]
FAMILY_BY_FEATURE = {f: group["id"] for group in FEATURE_GROUPS for f in group["feature_ids"]}
FAMILY_LABEL = {group["id"]: group["label"] for group in FEATURE_GROUPS}
CARDIO_IDS = [f for g in FEATURE_GROUPS if g["id"] in ("heart", "breathing") for f in g["feature_ids"] if not f.startswith("track.")]
FEATURE_SIGNAL = {f: group["signal"] for group in FEATURE_GROUPS for f in group["feature_ids"]}
assert sorted(TABLE_FEATURES) == sorted(WINDOW_FEATURES), "families must cover exactly the model's 19 features"

# Per stage: the features the explorer shows, and which end of each one the stage's example should come from.
STAGE_FEATURE_TARGETS = {
    "Wake": {"topography_features.occipital_rel_alpha": "high", "track.o_rel_alpha": "high", "track.c_rel_alpha": "high", "epoch_ratio_features.rel_beta": "high", "track.c_rel_beta": "high", "epoch_ratio_features.emg_floor_db": "high", "track.hr": "high"},
    "N1": {"epoch_ratio_features.rel_theta": "high", "track.c_rel_theta": "high", "spindle_features.aperiodic_exponent": "low", "epoch_ratio_features.eog_rem_vs_sem_db": "low", "breath_regularity.breath_interval_cv_60s": "high"},
    "N2": {"spindle_bursts": "high", "track.c_rel_sigma": "high", "cardiorespiratory_coupling.hr_resp_coherence_120s": "high"},
    "N3": {"topography_features.frontal_slow_wave_fraction": "high", "track.f_rel_delta": "high", "spindle_features.delta_power": "high", "track.c_log_power": "high", "hrv_features.hr_slow_sd_300s": "low"},
    "REM": {"track.c_log_power": "low", "epoch_ratio_features.rel_theta": "high", "epoch_ratio_features.eog_rem_vs_noise_db": "high", "eye_movement_activity.eye_movement_activity": "high", "eye_movement_activity.eye_movement_baseline": "high", "track.eog_high": "high", "epoch_ratio_features.emg_floor_db": "low", "cardiorespiratory_coupling.hr_resp_coherence_120s": "low", "track.resp_env": "low", "track.resp_effort": "high"},
}

# Model lineup: (id, name, role, question, source). Sources: (config, decoder) in the window run, or "cnn".
# Main page: A, B, C (adopted), D. The rest are intermediate steps shown in the supplement.
MODELS = [
    ("epoch", "A · Single epoch", "Reference: no context",
     "How far do the 19 features get when each 30-second epoch is classified on its own?", ("epoch19", "epoch")),
    ("epoch-hmm", "A + sleep-structure decoder", "Whole-night decoder",
     "How much does knowing how stages follow each other add when the model sees one epoch?", ("epoch19", "hmm")),
    ("window-features", "Features in context", "Intermediate: 6-s features",
     "The 19 features every 6 seconds over 2.5 minutes, before any one-second signal.", ("window19", "epoch")),
    ("window-all", "B · Context", "Context: tracks + features",
     "1-s tracks and 6-s features over 2.5 minutes, including heart and breathing.", ("window33", "epoch")),
    ("window", "27 signals", "Intermediate: no heart and breathing",
     "B without heart and breathing: EEG, eyes and chin only.", ("nocardio27", "epoch")),
    ("adopted", "C · 20 signals", "Adopted model",
     "B without heart and breathing and the least-used inputs, with a validated spindle-burst marker.", ("g20", "epoch")),
    ("cnn", "D · Attention CNN", "Neural comparison",
     "Does a network that learns its own filters over the 1-s signals beat boosting on fixed windows?", ("cnn", "epoch")),
]
ADOPTED = "adopted"
BASELINE = "epoch"


# ---------------------------------------------------------------- frame, folds and model metrics
def _prediction_frame() -> tuple[pd.DataFrame, list]:
    preds = pd.read_parquet(WINDOW_RUN / "predictions.parquet")
    preds["time_seconds"] = preds["epoch_index"].astype(float) * 30.0
    splits = list(build_subject_eval_splits(groups=preds["subject_id"].astype(str).to_numpy(),
                                            y=preds["label.stage_30s"].to_numpy(), outer_validation="kfold",
                                            outer_folds=5, random_state=7))
    preds["fold_index"] = -1
    for k, sp in enumerate(splits):
        preds.loc[np.asarray(sp.test_idx), "fold_index"] = k
    preds = preds.rename(columns={"label.stage_30s": "y_true"})
    return preds, splits


def _cnn_predictions(preds: pd.DataFrame, splits) -> tuple[np.ndarray, np.ndarray]:
    """Attention-CNN test probabilities (exploration run on the same frame and folds), decoded with the same HMM."""
    out = np.full(len(preds), -1)
    raw = np.full(len(preds), -1)
    y = preds["y_true"].to_numpy()
    subj = preds["subject_id"].astype(str).to_numpy()
    ei = preds["epoch_index"].to_numpy()

    def nights(idx):
        order = idx[np.lexsort((ei[idx], subj[idx]))]
        s = subj[order]
        bounds = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
        return [order[a:b] for a, b in zip(bounds[:-1], bounds[1:])]

    for k, sp in enumerate(splits):
        d = np.load(CNN_PROBS / f"probs_320_p2+16_c1_cnn_fold{k}.npz")
        test_idx = np.asarray(sp.test_idx)
        if not np.array_equal(np.sort(d["test_idx"]), np.sort(test_idx)):
            raise SystemExit(f"CNN fold {k} does not match the window run's test subjects")
        proba = np.zeros((len(preds), 5))
        proba[d["test_idx"]] = d["test"]
        raw[d["test_idx"]] = d["test"].argmax(1)
        tm = learn_transition_matrix([y[n] for n in nights(np.asarray(sp.train_idx))])
        for n in nights(test_idx):
            out[n] = posterior_decode(proba[n], tm).argmax(1)
    return raw, out


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")), "cohen_kappa": float(cohen_kappa_score(y, pred))}


def _model_results(preds: pd.DataFrame, columns: dict[str, str]) -> list[dict[str, Any]]:
    y = preds["y_true"].to_numpy()
    models = []
    for model_id, name, role, question, _ in MODELS:
        pred = preds[columns[model_id]].to_numpy()
        folds = []
        for k in range(5):
            m = preds["fold_index"].to_numpy() == k
            f = _metrics(y[m], pred[m])
            folds.append({"fold": k, "macro_f1": f["macro_f1"], "accuracy": f["accuracy"], "cohen_kappa": f["cohen_kappa"]})
        metrics = {key: float(np.mean([f[key] for f in folds])) for key in ("macro_f1", "accuracy", "cohen_kappa")}
        metrics["balanced_accuracy"] = float(np.mean([balanced_accuracy_score(y[preds["fold_index"] == k], pred[preds["fold_index"] == k]) for k in range(5)]))
        models.append({"id": model_id, "name": name, "role": role, "hypothesis": question, "metrics": metrics, "folds": folds,
                       "stage_f1": dict(zip(STAGES, map(float, f1_score(y, pred, labels=range(5), average=None))))})
    return models


def _ahi_bands() -> dict[str, str]:
    info = pd.concat([pd.read_csv(path) for path in SUBJECT_INFO if path.exists()], ignore_index=True)
    return dict(zip(info["record_id"].astype(str), info["ahi_band"].astype(str)))


def _diagnostics(preds: pd.DataFrame, column: str) -> dict[str, Any]:
    y_true = preds["y_true"].astype(int).to_numpy()
    y_pred = preds[column].astype(int).to_numpy()
    counts = confusion_matrix(y_true, y_pred, labels=range(5))
    normalized = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(5), zero_division=0)
    subject_metrics = []
    ahi = _ahi_bands()
    for subject_id, g in preds.groupby("subject_id", sort=True):
        t, p = g["y_true"].astype(int).to_numpy(), g[column].astype(int).to_numpy()
        subject_metrics.append({
            "subject": str(subject_id), "epochs": int(len(g)), "accuracy": float(accuracy_score(t, p)),
            "macro_f1": float(f1_score(t, p, labels=range(5), average="macro", zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(t, p)),
            "ahi_band": ahi.get(str(subject_id), "unknown"),
        })
    return {
        "rows": int(len(preds)),
        "subjects": int(preds["subject_id"].nunique()),
        "confusion_counts": counts.astype(int).tolist(),
        "confusion_normalized": normalized.tolist(),
        "per_stage": [{"stage": STAGES[i], "precision": float(precision[i]), "recall": float(recall[i]),
                       "f1": float(f1[i]), "support": int(support[i])} for i in range(5)],
        "subject_metrics": subject_metrics,
    }


def _validation_folds(preds: pd.DataFrame) -> list[dict[str, Any]]:
    all_subjects = sorted(preds["subject_id"].astype(str).unique())
    folds = []
    for k, g in preds.groupby("fold_index", sort=True):
        test = sorted(g["subject_id"].astype(str).unique())
        folds.append({"fold": int(k), "train_subjects": [s for s in all_subjects if s not in test], "test_subjects": test})
    return folds


# ---------------------------------------------------------------- importance and sleep structure
def _signal_id(name: str) -> str:
    """Importance rows name signals by short name; map back to explorer ids."""
    for f in FEATURE_IDS:
        if f.split(".")[-1] == name:
            return f
    raise KeyError(name)


def _importance() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    imp = pd.read_csv(WINDOW_RUN / "importance.csv")
    agg = imp.groupby(["group", "name"], sort=False).mean(numeric_only=True).drop(columns="fold").reset_index()
    n_folds = int(imp["fold"].nunique())
    signals = []
    for _, row in agg[agg["group"] == "signal"].sort_values("drop", ascending=False).iterrows():
        fid = _signal_id(str(row["name"]))
        signals.append({"id": fid, "label": FEATURE_LABELS[fid], "family_id": FAMILY_BY_FEATURE[fid],
                        "family": FAMILY_LABEL[FAMILY_BY_FEATURE[fid]],
                        "kind": "track" if fid.startswith("track.") else "feature",
                        "importance": float(row["drop"]), "stage_drops": {s: float(row[f"drop_{s}"]) for s in STAGES}})
    steps = [{"offset_sec": int(str(row["name"]).rstrip("s")), "importance": float(row["drop"]),
              "stage_drops": {s: float(row[f"drop_{s}"]) for s in STAGES}}
             for _, row in agg[agg["group"] == "step"].iterrows()]
    families = []
    for group in FEATURE_GROUPS:
        members = [s for s in signals if s["family_id"] == group["id"]]
        if not members:  # a family the adopted model does not use
            continue
        families.append({"group": group["label"], "family_id": group["id"], "n_features": len(members),
                         "importance": float(sum(s["importance"] for s in members)),
                         "stage_drops": {st: float(sum(s["stage_drops"][st] for s in members)) for st in STAGES}})
    families.sort(key=lambda f: f["importance"], reverse=True)
    return signals, steps, families, n_folds


def _time_importance() -> dict[str, Any]:
    """Fine-scale signals shuffled per 6-s step, epoch-level features per epoch (scripts/run_window_time_importance.py)."""
    t = pd.read_csv(WINDOW_RUN / "time_importance.csv")
    agg = t.groupby(["group", "offset_sec", "span_sec"], as_index=False)["drop"].mean()
    rows = lambda g: [{"offset_sec": int(r.offset_sec), "span_sec": int(r.span_sec), "importance": float(r.drop)}  # noqa: E731
                      for r in agg[agg["group"] == g].sort_values("offset_sec").itertuples()]
    return {"fine": rows("fine"), "epoch": rows("epoch"), "folds": int(t["fold"].nunique())}


def _generalisation() -> dict[str, Any]:
    """Cross-lab transfer (run_window_transfer.py) and the final held-out test (run_window_holdout.py), when present."""
    out: dict[str, Any] = {"transfer": [], "holdout": [], "holdout_subjects": 0, "holdout_looks": 0, "holdout_dates": []}
    path = ARTIFACTS / "staging_window_model" / "transfer" / "summary.csv"
    if path.exists():
        out["transfer"] = [{"evaluation": str(r.evaluation), "macro_f1": float(r.macro_f1), "kappa": float(r.kappa),
                            "accuracy": float(r.accuracy), "subj_min": float(r.subj_min), "subj_median": float(r.subj_median),
                            "stage_f1": {st: float(getattr(r, f"f1_{st}")) for st in STAGES}} for r in pd.read_csv(path).itertuples()]
    path = WINDOW_RUN / "holdout" / "summary.csv"
    if path.exists():  # headline metrics only, by design; look 1 scored the 27-signal model and references
        looks = [(1, WINDOW_RUN / "holdout" / "summary_look1.csv"), (2, path)]
        seen = set()
        rows = []
        for look, f in looks:
            if f.exists():
                for r in pd.read_csv(f).itertuples():
                    if (str(r.config), look) in seen or (look == 2 and str(r.config) != "g20"):
                        continue
                    seen.add((str(r.config), look))
                    rows.append({"config": str(r.config), "look": look, "macro_f1": float(r.macro_f1), "kappa": float(r.kappa),
                                 "accuracy": float(r.accuracy)})
        out["holdout"] = rows
        ledger = pd.read_csv(WINDOW_RUN / "holdout" / "LEDGER.csv")
        out["holdout_subjects"] = int(ledger["subjects"].iloc[-1])
        out["holdout_looks"] = int(ledger["look"].nunique())
        out["holdout_dates"] = [str(d) for d in ledger.drop_duplicates("look", keep="last")["date"]]
    return out


def _reduction() -> list[dict[str, Any]]:
    """Feature reduction and swap rounds (staging_window_model/cinc2018/feature_reduction/summary.csv)."""
    path = WINDOW_RUN / "feature_reduction" / "summary.csv"
    if not path.exists():
        return []
    return [{"config": str(r.config), "signals": int(r.n_signals), "macro_f1": float(r.macro_f1), "folds": str(r.folds),
             "n2_f1": float(r.f1_N2)} for r in pd.read_csv(path).itertuples()]


def _complexity() -> list[dict[str, Any]]:
    """The window model with simpler learners (scripts/run_window_complexity.py)."""
    path = WINDOW_RUN / "complexity" / "summary.csv"
    if not path.exists():
        return []
    labels = {"hgb_l31_i100": "Default boosting (adopted)", "hgb_l8_i100": "Boosting, 8-leaf trees",
              "hgb_l4_i100": "Boosting, 4-leaf trees", "hgb_l2_i200": "Boosted single splits (additive)",
              "tree_d4": "One tree, depth 4", "tree_d6": "One tree, depth 6", "tree_d8": "One tree, depth 8"}
    rows = pd.read_csv(path)
    return [{"learner": str(r.learner), "label": labels.get(str(r.learner), str(r.learner)), "macro_f1": float(r.macro_f1),
             "kappa": float(r.kappa), "learned_values": int(r.learned_values), "folds": str(r.folds),
             "stage_f1": {st: float(getattr(r, f"f1_{st}")) for st in STAGES}} for r in rows.itertuples()]


def _signal_ablations() -> list[dict[str, Any]]:
    """Window model with signal groups removed or isolated (run_window_hmm_staging.py --tag ablations / cardio_full*).

    Adds 4-class kappa (Wake / light N1+N2 / deep N3 / REM), the metric published heart-rate staging models report.
    """
    labels = {"window33": "All 33 signals, with heart and breathing", "nocardio27": "EEG, eyes and chin only (27 signals)", "cardio6": "Heart and breathing, 6 signals, 2.5 min",
              "cardio6_c5": "Heart and breathing, 6 signals, 5.5 min",
              "cardiofull_c5": "Heart and breathing, all 12 signals, 5.5 min",
              "cardiofull_c10": "Heart and breathing, all 12 signals, 10.5 min"}
    four = np.array([0, 1, 1, 2, 3])
    out = []
    for tag, suffix in (("", ""), ("ablations", ""), ("cardio_full", ""), ("cardio_full_unbalanced", ", unbalanced")):
        folder = WINDOW_RUN / tag if tag else WINDOW_RUN
        if not (folder / "summary.csv").exists():
            continue
        rows = pd.read_csv(folder / "summary.csv")
        if not tag:
            rows = rows[rows["config"] == "window33"]
        preds = pd.read_parquet(folder / "predictions.parquet")
        y4 = four[preds["label.stage_30s"].to_numpy()]
        for r in rows.itertuples():
            col = f"{r.config}__{'epoch' if r.decoder == 'epoch-only' else 'hmm'}"
            kappa4 = float(cohen_kappa_score(y4, four[preds[col].to_numpy()])) if col in preds else float("nan")
            out.append({"config": f"{r.config}{'_unbalanced' if suffix else ''}", "label": labels.get(str(r.config), str(r.config)) + suffix,
                        "decoder": str(r.decoder), "macro_f1": float(r.macro_f1), "kappa": float(r.kappa), "kappa4": kappa4,
                        "stage_f1": {st: float(getattr(r, f"f1_{st}")) for st in STAGES}})
    return out


def _sleep_structure() -> dict[str, Any]:
    trans = pd.read_csv(WINDOW_RUN / "transitions.csv", index_col=0)
    bouts = pd.read_csv(WINDOW_RUN / "bouts.csv")
    return {
        "transition_matrix": trans.loc[STAGES, STAGES].to_numpy().tolist(),
        "bouts": [{k: (v if isinstance(v, str) else float(v)) for k, v in row.items()} for row in bouts.to_dict("records")],
    }


# ---------------------------------------------------------------- feature distributions and examples
def _track_epoch_means(subject_id: str) -> pd.DataFrame:
    t = pd.read_parquet(TRACK_CACHE / f"{subject_id}.parquet")
    names = [f.split(".", 1)[1] for f in TRACK_FEATURES]
    n = len(t) // 30
    vals = t[names].to_numpy(float)[: n * 30].reshape(n, 30, len(names))
    with np.errstate(all="ignore"):
        import warnings

        warnings.simplefilter("ignore")
        means = np.nanmean(vals, axis=1)
    out = pd.DataFrame(means, columns=TRACK_FEATURES)
    out["time_seconds"] = np.arange(n) * 30.0
    return out


def _feature_matrix(preds: pd.DataFrame, column: str) -> pd.DataFrame:
    base = preds[["subject_id", "time_seconds", "y_true", column]].rename(columns={column: "y_pred"})
    rows = []
    for subject_id, sp in base.groupby("subject_id", sort=True):
        ft = pd.read_parquet(RECIPE_CACHE / str(subject_id) / "feature_table_py" / "artifacts" / "feature_table.parquet",
                             columns=["time_seconds", *TABLE_FEATURES])
        ft["time_seconds"] = np.floor(ft["time_seconds"].astype(float) / 30.0) * 30.0
        epoch_features = ft.replace([np.inf, -np.inf], np.nan).groupby("time_seconds", as_index=False)[TABLE_FEATURES].median()
        merged = sp.merge(epoch_features, on="time_seconds", how="left").merge(_track_epoch_means(str(subject_id)), on="time_seconds", how="left")
        bursts = spindle_burst_counts(pd.read_parquet(TRACK_CACHE / f"{subject_id}.parquet"))
        merged = merged.merge(pd.DataFrame({"time_seconds": np.arange(len(bursts)) * 30.0, SPINDLE_BURSTS: bursts}), on="time_seconds", how="left")
        rows.append(merged)
    merged = pd.concat(rows, ignore_index=True).sort_values(["subject_id", "time_seconds"]).reset_index(drop=True)
    g = merged.groupby("subject_id", sort=False)
    merged["stable_stage"] = (
        g["y_true"].shift(1).eq(merged["y_true"]) & g["y_true"].shift(-1).eq(merged["y_true"])
        & (merged["time_seconds"] - g["time_seconds"].shift(1)).eq(30.0)
        & (g["time_seconds"].shift(-1) - merged["time_seconds"]).eq(30.0)
    )
    return merged


def _cohens_d(target: pd.Series, rest: pd.Series) -> float:
    a, b = target.dropna().to_numpy(float), rest.dropna().to_numpy(float)
    if a.size < 2 or b.size < 2:
        return 0.0
    pooled = ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / (a.size + b.size - 2)
    return float((a.mean() - b.mean()) / np.sqrt(pooled)) if np.isfinite(pooled) and pooled > 0 else 0.0


def _feature_distributions(merged: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for fid in FEATURE_IDS:
        stage_rows = []
        for k, stage in enumerate(STAGES):
            values = merged.loc[merged["y_true"] == k, fid].dropna()
            rest = merged.loc[merged["y_true"] != k, fid].dropna()
            q = values.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            stage_rows.append({"stage": stage, "count": int(len(values)), "p10": float(q.loc[0.10]), "p25": float(q.loc[0.25]),
                               "median": float(q.loc[0.50]), "p75": float(q.loc[0.75]), "p90": float(q.loc[0.90]),
                               "one_vs_rest_cohens_d": _cohens_d(values, rest)})
        out.append({"id": fid, "family_id": FAMILY_BY_FEATURE[fid], "label": FEATURE_LABELS[fid],
                    "signal": FEATURE_SIGNAL[fid], "stages": stage_rows})
    return out


CINC_CHANNELS = {
    "central": (["C3-M2"], "EEG C3–M2"),
    "eog": (["E1-M2"], "EOG E1–M2"),
    "emg": (["Chin1-Chin2"], "Chin EMG"),
    "effort": (["ABD", "CHEST"], "Respiratory effort (abdomen + chest), 60 s"),
    "frontal": (["F4-M1"], "EEG F4–M1 (µV)"),
    "occipital": (["O1-M2"], "EEG O1–M2"),
}


def _raw_channel_for(feature_id: str) -> str:
    name = feature_id.split(".")[-1]
    if feature_id in ("topography_features.frontal_slow_wave_fraction", "track.f_rel_delta"):
        return "frontal"
    if feature_id in ("topography_features.occipital_rel_alpha", "track.o_rel_alpha"):
        return "occipital"
    if feature_id.startswith("eye_movement_activity") or "eog" in name:
        return "eog"
    if "emg" in name:
        return "emg"
    if feature_id.startswith(("breath_regularity", "cardiorespiratory_coupling")) or name.startswith("resp_"):
        return "effort"
    if feature_id.startswith("hrv_features") or name == "hr":
        return "heart_rate"
    return "central"


def _heart_rate_excerpt(record_id: str, centre_sec: float, half_window: float = 150.0) -> dict[str, Any]:
    peaks = pd.read_parquet(RECIPE_CACHE / record_id / "pan_tompkins_py" / "events" / "r_peaks.parquet",
                            columns=["start"])["start"].to_numpy(float) / ECG_FS
    rr, beat_time = np.diff(peaks), peaks[1:]
    bpm = 60.0 / rr
    keep = (bpm >= 30) & (bpm <= 200) & (beat_time >= centre_sec - half_window) & (beat_time <= centre_sec + half_window)
    grid = np.arange(centre_sec - half_window, centre_sec + half_window, 0.5)
    values = np.interp(grid, beat_time[keep], bpm[keep]) if keep.sum() >= 2 else np.full(grid.size, np.nan)
    return {"id": "heart_rate", "label": "Heart rate, 5 min (bpm)", "start_sec": float(grid[0]), "end_sec": float(grid[-1]),
            "effective_sample_rate_hz": 2.0, "values": np.round(values - np.nanmedian(values), 3).tolist()}


def _cinc_excerpt(record_id: str, start_sec: float, end_sec: float, feature_id: str) -> dict[str, Any]:
    import wfdb

    signal_id = _raw_channel_for(feature_id)
    if signal_id == "heart_rate":
        return _heart_rate_excerpt(record_id, (start_sec + end_sec) / 2.0)
    if signal_id == "effort":
        centre = (start_sec + end_sec) / 2.0
        start_sec, end_sec = centre - 30.0, centre + 30.0
    names, label = CINC_CHANNELS[signal_id]
    fs = 200.0
    record = wfdb.rdrecord(str(ROOT / "data" / "raw" / "cinc2018" / record_id / record_id),
                           sampfrom=max(int(start_sec * fs), 0), sampto=int(end_sec * fs), channel_names=names)
    values = record.p_signal.sum(axis=1)
    stride = max(int(math.ceil(values.size / 900)), 1)
    sampled = values[::stride]
    return {"id": signal_id, "label": label, "start_sec": start_sec, "end_sec": end_sec,
            "effective_sample_rate_hz": fs / stride, "values": np.round(sampled - np.nanmedian(sampled), 5).tolist()}


def _signal_is_usable(signal: dict[str, Any]) -> bool:
    values = np.asarray(signal["values"], dtype=float)
    finite = values[np.isfinite(values)]
    # An atonic chin sits near the amplifier's resolution, so its trace has few distinct levels; that flatness is the point.
    min_levels = 10 if signal.get("id") == "emg" else 50
    if finite.size < 100 or np.unique(np.round(finite, 6)).size < min_levels:
        return False
    q01, q25, q75, q99 = np.quantile(finite, [0.01, 0.25, 0.75, 0.99])
    span = q75 - q25
    if signal.get("id") == "emg":
        return bool(q99 - q01 > 0)
    return bool(span > 0 and (q99 - q01) / span < 40)


def _representative_signal_examples(merged: pd.DataFrame) -> list[dict[str, Any]]:
    output = []
    for target_stage, targets in STAGE_FEATURE_TARGETS.items():
        target_index = STAGES.index(target_stage)
        for feature_id, direction in targets.items():
            lo, hi = merged[feature_id].dropna().quantile([0.01, 0.99])
            eligible = merged.loc[merged["stable_stage"] & merged["y_true"].eq(merged["y_pred"])
                                  & merged[feature_id].between(lo, hi)].copy()
            # Typical rather than extreme cases: the target near its stage's 80th percentile in the feature's direction;
            # the contrast near the median of the stage that sits furthest the other way.
            medians = eligible.groupby("y_true")[feature_id].median()
            others = medians.drop(target_index, errors="ignore")
            contrast_index = int(others.idxmin() if direction == "high" else others.idxmax())
            for role, candidates in (("target", eligible[eligible["y_true"].eq(target_index)].copy()),
                                     ("contrast", eligible[eligible["y_true"].eq(contrast_index)].copy())):
                candidates["selection_percentile"] = candidates[feature_id].rank(pct=True, ascending=True)
                q = (0.8 if direction == "high" else 0.2) if role == "target" else 0.5
                aim = candidates[feature_id].quantile(q)
                candidates = candidates.assign(_gap=(candidates[feature_id] - aim).abs()).sort_values(["_gap", "subject_id", "time_seconds"])
                chosen = None
                # Low-amplitude channels can sit at the amplifier's quantization floor; search deeper before giving up.
                for row in candidates.head(400).itertuples(index=False):
                    signal = _cinc_excerpt(str(row.subject_id), float(row.time_seconds), float(row.time_seconds) + 30.0, feature_id)
                    if _signal_is_usable(signal):
                        chosen = (row, signal)
                        break
                if chosen is None:
                    print(f"  warning: no usable {role} example for {target_stage} / {feature_id}; skipped")
                    continue
                row, signal = chosen
                value = float(merged.loc[merged["subject_id"].eq(row.subject_id)
                                         & merged["time_seconds"].eq(row.time_seconds), feature_id].iloc[0])
                population = merged[feature_id].dropna().to_numpy(float)
                stage = STAGES[int(row.y_true)]
                output.append({
                    "id": f"{target_stage.lower()}::{feature_id}::{role}",
                    "family_id": FAMILY_BY_FEATURE[feature_id],
                    "target_stage": target_stage,
                    "stage": stage,
                    "record_id": str(row.subject_id),
                    "epoch_start_sec": float(row.time_seconds),
                    "feature_id": feature_id,
                    "feature_label": FEATURE_LABELS[feature_id],
                    "feature_value": value,
                    "feature_percentile": float(np.mean(population <= value) * 100.0),
                    "stage_percentile": float(row.selection_percentile * 100.0),
                    "direction": direction,
                    "comparison_role": role,
                    "selection_note": (f"Typical clear {target_stage} epoch (80th percentile of {target_stage})."
                                       if role == "target" else f"Typical {stage} epoch (median of {stage})."),
                    "signal": signal,
                })
    return output


# ---------------------------------------------------------------- main
def main() -> int:
    preds, splits = _prediction_frame()
    preds["cnn__epoch"], preds["cnn__hmm"] = _cnn_predictions(preds, splits)
    columns = {model_id: f"{src[0]}__{src[1]}" for model_id, *_, src in MODELS}
    models = _model_results(preds, columns)
    diagnostics = _diagnostics(preds, columns[ADOPTED])
    signal_importance, step_importance, family_importance, n_imp_folds = _importance()
    feature_matrix = _feature_matrix(preds, columns[ADOPTED])
    examples = _representative_signal_examples(feature_matrix)
    baseline = next(m for m in models if m["id"] == BASELINE)
    adopted = next(m for m in models if m["id"] == ADOPTED)

    payload = {
        "source": {
            "dataset": DATASET_NAME,
            "short_name": "CinC 2018",
            "subjects": diagnostics["subjects"],
            "epochs": diagnostics["rows"],
            "epoch_seconds": 30,
            # The adopted model's inputs (no heart or breathing); the explorer also shows those families.
            "features": 12,  # model C: 11 table features + spindle bursts
            "tracks": 8,  # model C: 8 one-second tracks
            "stages": STAGES,
        },
        "validation": {
            "outer_folds": 5,
            "train_subjects_per_fold": diagnostics["subjects"] - diagnostics["subjects"] // 5,
            "test_subjects_per_fold": diagnostics["subjects"] // 5,
            "grouping": "subject",
            "primary_metric": "Macro-F1",
            "folds": _validation_folds(preds),
        },
        "headline": {
            "baseline_macro_f1": baseline["metrics"]["macro_f1"],
            "best_model_id": ADOPTED,
            "best_macro_f1": adopted["metrics"]["macro_f1"],
            "diagnostics_model_id": ADOPTED,
            "absolute_lift": adopted["metrics"]["macro_f1"] - baseline["metrics"]["macro_f1"],
        },
        "models": models,
        "diagnostics": diagnostics,
        "feature_importance": family_importance,
        "signal_importance": signal_importance,
        "step_importance": step_importance,
        "importance_folds": n_imp_folds,
        # From scripts/count_window_model_size.py: fold 0 window model refit once, trees/splits/leaves counted.
        "model_size": json.loads((WINDOW_RUN / "model_size.json").read_text(encoding="utf-8")),
        "time_importance": _time_importance(),
        "sleep_structure": _sleep_structure(),
        "signal_ablations": _signal_ablations(),
        "complexity": _complexity(),
        "generalisation": _generalisation(),
        "feature_evidence": {"distributions": _feature_distributions(feature_matrix)},
        "reduction": _reduction(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    for m in models:
        print(f"  {m['id']:<22} macro-F1 {m['metrics']['macro_f1']:.4f}  kappa {m['metrics']['cohen_kappa']:.3f}  "
              f"folds {' '.join(f'{f['macro_f1']:.3f}' for f in m['folds'])}")
    FEATURE_EXAMPLES_OUTPUT.write_text(json.dumps(examples, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {FEATURE_EXAMPLES_OUTPUT} ({len(examples)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
