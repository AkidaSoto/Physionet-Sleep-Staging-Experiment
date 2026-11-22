"""Export Model D (#79 Concept-BiGRU) internal, external (holdout + ISRUC), conformal & concept JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "showcase" / "content" / "generated"
OUT = GENERATED / "concept-showcase.json"
STAGES = ["Wake", "N1", "N2", "N3", "REM"]
PROBS = [f"p_{s}" for s in STAGES]

CONCEPT_META = {
    "slow_wave": "Slow-wave & delta",
    "n2_events": "N2 spindles & K-complexes",
    "wake_n1": "Wake / N1 arousal",
    "rem_tone": "REM ocular & chin tone",
    "autonomic": "Autonomic (HR + breathing)",
}

CUE_META = {
    "c_rel_delta": ("Central relative delta (0.5–4 Hz)", "slow_wave"),
    "f_rel_delta": ("Frontal relative delta (0.5–4 Hz)", "slow_wave"),
    "eog_low": ("Slow rolling EOG (0.5–2 Hz)", "rem_tone"),
    "eog_burst_guarded": ("Guarded EOG bursts (F3/F4 veto)", "rem_tone"),
    "emg_log_rms": ("Chin EMG tone (10–80 Hz)", "rem_tone"),
    "hr": ("Heart rate (ECG R-peaks)", "autonomic"),
    "resp_env": ("Respiratory envelope amplitude", "autonomic"),
    "resp_effort": ("Raw respiratory effort belt", "rejected"),
}


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _select_median_night(
    df: pd.DataFrame,
    summary: dict,
    *,
    cohort_label: str,
    model_dir: Path | None = None,
) -> dict:
    all_f1s = []
    candidates = []
    for sub, grp in df.groupby("subject_id", sort=False):
        y_s = grp["label.stage_30s"].to_numpy()
        p_s = grp["prediction"].to_numpy()
        present_all = sorted(set(y_s.tolist()))
        present_5 = [i for i in range(5) if (y_s == i).sum() >= 5]
        f1 = float(f1_score(y_s, p_s, labels=present_all, average="macro", zero_division=0))
        all_f1s.append(f1)
        fold_val = int(grp["fold"].iloc[0]) if "fold" in grp.columns else -1
        # Require all present stages to have >= 5 epochs and at least 4 stages present
        # so the hypnogram has no 1-epoch outlier stage and matches Figure 2's >=1-epoch score identically.
        if len(present_all) >= 4 and len(present_all) == len(present_5):
            candidates.append((str(sub), fold_val, len(grp), f1))
    cohort_med = float(np.median(all_f1s))
    candidates.sort(key=lambda row: (abs(row[3] - cohort_med), row[0]))
    med_sub, med_fold, _, med_f1 = candidates[0]

    night = (
        df[df.subject_id.astype(str) == med_sub]
        .sort_values("epoch_index")
        .reset_index(drop=True)
    )
    probs_n = night[PROBS].to_numpy()
    pred_n = night["prediction"].to_numpy()
    expert_n = night["label.stage_30s"].to_numpy()

    conf = summary["conformal"]["calibrated"]["coverage_90"]["mondrian"]["folds"]
    if med_fold >= 0:
        thr = np.asarray(next(row["threshold"] for row in conf if row["fold"] == med_fold), dtype=np.float64)
    else:
        thr = np.mean([row["threshold"] for row in conf], axis=0)
    sets = (1.0 - probs_n) <= thr[None, :]
    empty = ~sets.any(axis=1)
    if empty.any():
        sets[empty, probs_n[empty].argmax(axis=1)] = True
    set_sizes = sets.sum(axis=1).astype(int)

    concept_names = list(summary["concept_dims"].keys())
    if f"concept_{concept_names[0]}_Wake" in night.columns:
        c_night = night
    else:
        assert model_dir is not None
        c_df = pd.read_parquet(model_dir / f"fold-{med_fold}" / "concept_logits.parquet")
        c_night = (
            c_df[c_df.subject_id.astype(str) == med_sub]
            .sort_values("epoch_index")
            .reset_index(drop=True)
        )
    ep_idx = np.arange(len(night))
    # Center each (concept, stage) logit across the night to remove static GELU(LayerNorm) bias
    # before selecting the top elevated concept driver for the predicted stage at each epoch.
    centered_cols = []
    for c in concept_names:
        mat = np.column_stack([c_night[f"concept_{c}_{s}"].to_numpy() for s in STAGES])
        mat = mat - mat.mean(axis=0, keepdims=True)
        centered_cols.append(mat[ep_idx, pred_n])
    contrib = np.column_stack(centered_cols)
    top_concept = contrib.argmax(axis=1).astype(int)

    return {
        "cohort": cohort_label,
        "record": med_sub,
        "fold": med_fold,
        "epochs": int(len(night)),
        "macro_f1": round(med_f1, 4),
        "disagreements": int((expert_n != pred_n).sum()),
        "concept_labels": [CONCEPT_META.get(c, c) for c in concept_names],
        "expert": [int(v) for v in expert_n],
        "model": [int(v) for v in pred_n],
        "set_sizes_90": [int(v) for v in set_sizes],
        "top_concept": [int(v) for v in top_concept],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--active-exp", type=int, choices=[79, 80], default=79)
    args = ap.parse_args()

    s75 = _load_json(ROOT / "artifacts/exploration/cinc-bigru-unified/summary.json")
    s76 = _load_json(ROOT / "artifacts/exploration/cinc-hgb-704/summary.json")
    s78 = _load_json(ROOT / "artifacts/exploration/cinc-unified-ensemble/summary.json")
    s79 = _load_json(ROOT / "artifacts/exploration/cinc-bigru-concept/summary.json")
    s80_dir = ROOT / "artifacts/exploration/cinc-bigru-cues/rem_auto"
    s80 = _load_json(s80_dir / "summary.json")
    if s80 is not None and "calibrated" not in s80:
        s80 = None
    d_screen = _load_json(ROOT / "artifacts/exploration/cinc-bigru-cues/cohen_d_screen.json")
    dash = _load_json(GENERATED / "staging-dashboard.json")
    lead_old = _load_json(GENERATED / "lead-figure.json")

    ho_sum = _load_json(ROOT / "artifacts/exploration/cinc-bigru-concept/external_eval/holdout_summary.json")
    is_sum = _load_json(ROOT / "artifacts/exploration/cinc-bigru-concept/external_eval/isruc_summary.json")

    active_exp = 80 if (args.active_exp == 80 and s80 is not None) else 79
    active_summary = s80 if active_exp == 80 else s79
    active_dir = s80_dir if active_exp == 80 else (ROOT / "artifacts/exploration/cinc-bigru-concept")
    assert active_summary is not None

    # Clean A -> B -> C -> D main progression ladder (tangents live in Supplementary)
    old_models = {m["id"]: m for m in dash["models"]} if dash else {}
    ladder = []
    if "epoch" in old_models:
        m = old_models["epoch"]
        ladder.append({
            "id": "epoch",
            "label": "A · Single-epoch trees",
            "exp": 16,
            "kind": "Tabular · 19 epoch features",
            "macro_f1": round(m["metrics"]["macro_f1"], 4),
            "kappa": round(m["metrics"]["cohen_kappa"], 4),
            "folds": [round(f["macro_f1"], 4) for f in m["folds"]],
            "size": "29k tree values",
            "train_patients": "255 / fold",
        })
    if "window-all" in old_models:
        m = old_models["window-all"]
        ladder.append({
            "id": "window-all",
            "label": "B · 2.5-min context trees",
            "exp": 23,
            "kind": "Tabular · 33 signals, 150 s",
            "macro_f1": round(m["metrics"]["macro_f1"], 4),
            "kappa": round(m["metrics"]["cohen_kappa"], 4),
            "folds": [round(f["macro_f1"], 4) for f in m["folds"]],
            "size": "29k tree values",
            "train_patients": "255 / fold",
        })
    if "adopted" in old_models:
        m = old_models["adopted"]
        ladder.append({
            "id": "adopted",
            "label": "C · Compact window trees",
            "exp": 36,
            "kind": "Tabular · 20 signals, 150 s",
            "macro_f1": round(m["metrics"]["macro_f1"], 4),
            "kappa": round(m["metrics"]["cohen_kappa"], 4),
            "folds": [round(f["macro_f1"], 4) for f in m["folds"]],
            "size": "29k tree values",
            "train_patients": "255 / fold",
        })
    cal_act = active_summary["calibrated"]
    ladder.append({
        "id": "model_d",
        "label": "D · Concept-BiGRU (#79)",
        "exp": active_exp,
        "kind": "4 additive concepts + 1-layer BiGRU + conformal",
        "macro_f1": round(cal_act["mean_fold_macro_f1"], 4),
        "kappa": round(cal_act["kappa"], 4),
        "folds": [round(x, 4) for x in cal_act["fold_macro_f1"]],
        "size": f"{round(active_summary['model_params'] / 1000, 1)}k weights",
        "train_patients": "704 / fold",
    })

    # Supplementary comparators (#75 Dense BiGRU, #76 Multi-Scale HGB, #78 Ensemble, #80 Cues)
    supp_models = []
    if s75 is not None:
        c75 = s75["calibrated"]
        supp_models.append({
            "id": "bigru75",
            "label": "#75 · Dense Unified BiGRU",
            "macro_f1": round(c75["mean_fold_macro_f1"], 4),
            "kappa": round(c75["kappa"], 4),
            "size": "266k weights (2 layers, 120 raw dims)",
        })
    if s76 is not None:
        c76 = s76["calibrated"]
        supp_models.append({
            "id": "hgb704",
            "label": "#76 · Multi-scale HGB (704 pts)",
            "macro_f1": round(c76["macro_f1"], 4),
            "kappa": round(c76["kappa"], 4),
            "size": "309 multi-scale features",
        })
    if s80 is not None:
        c80 = s80["calibrated"]
        supp_models.append({
            "id": "cues80",
            "label": "#80 · Concept-BiGRU + EMG/HR/EOG cues",
            "macro_f1": round(c80["mean_fold_macro_f1"], 4),
            "kappa": round(c80["kappa"], 4),
            "size": f"{round(s80['model_params'] / 1000, 1)}k weights (5 concepts, log loss {c80['log_loss']:.3f})",
        })
    if s78 is not None:
        b78 = s78["ensembles"][s78["best_ensemble"]]
        supp_models.append({
            "id": "ensemble78",
            "label": "#78 · Multi-family log-odds ensemble",
            "macro_f1": round(b78["macro_f1"], 4),
            "kappa": round(b78["kappa"], 4),
            "size": "6-model blend ceiling",
        })

    # Concept attributions & gates
    cd = active_summary["concept_diagnostics"]
    concept_widths = active_summary.get("concept_widths") or {
        k: active_summary["config"].get("width_per_concept", 16)
        for k in active_summary["concept_dims"]
    }
    concepts = []
    for c_name, ab in cd["concept_ablation"].items():
        concepts.append({
            "id": c_name,
            "label": CONCEPT_META.get(c_name, c_name),
            "dims": int(active_summary["concept_dims"][c_name]),
            "width": int(concept_widths[c_name]),
            "local_macro_f1_drop": round(float(ab["local_macro_f1_drop"]), 4),
            "full_direct_drop": round(float(ab["full_direct_drop"]), 4),
            "per_stage_local_f1_delta": ab["per_stage_local_f1_delta"],
            "mean_abs_true_stage_logit": ab["mean_abs_true_stage_logit"],
        })
    concepts.sort(key=lambda x: x["local_macro_f1_drop"])

    gates = [
        {
            "stage": st,
            "past": round(float(cd["stage_gates"][st]["past_gate_mean"]), 4),
            "future": round(float(cd["stage_gates"][st]["future_gate_mean"]), 4),
        }
        for st in STAGES
    ]

    cues_table = []
    if d_screen is not None:
        for cue, vals in d_screen.items():
            best_c, best_d = max(vals["mean_cohen_d"].items(), key=lambda kv: abs(kv[1]))
            label, branch = CUE_META.get(cue, (cue, "unknown"))
            cues_table.append({
                "id": cue,
                "label": label,
                "branch": branch,
                "qualified": bool(vals["qualified"]),
                "max_abs_d": round(float(vals["max_abs_d"]), 3),
                "best_contrast": best_c.replace("_", " "),
                "best_d": round(float(best_d), 3),
            })
        cues_table.sort(key=lambda x: x["max_abs_d"], reverse=True)

    cv_preds_df = pd.read_parquet(active_dir / "calibrated_predictions.parquet")
    cv_y = cv_preds_df["label.stage_30s"].to_numpy()
    cv_p = cv_preds_df["prediction"].to_numpy()
    cv_acc = float((cv_y == cv_p).mean())
    cv_subj_f1s = [
        float(f1_score(g["label.stage_30s"], g["prediction"], labels=sorted(set(g["label.stage_30s"].tolist())), average="macro", zero_division=0))
        for _, g in cv_preds_df.groupby("subject_id", sort=False)
    ]
    cv_subj_median = float(np.median(cv_subj_f1s))

    # Prefer #79's median held-out night for Figure 1 once external_eval exists!
    ho_preds_path = active_dir / "external_eval" / "holdout_predictions.parquet"
    if active_exp == 79 and ho_preds_path.is_file():
        night_trace = _select_median_night(
            pd.read_parquet(ho_preds_path),
            active_summary,
            cohort_label="holdout",
        )
    else:
        night_trace = _select_median_night(
            cv_preds_df,
            active_summary,
            cohort_label="cv",
            model_dir=active_dir,
        )

    external_eval = None
    if ho_sum is not None and is_sum is not None and active_exp == 79:
        coimbra_scorers = sorted(lead_old["patients"]["coimbra_scorers"]) if lead_old else []
        ho_vals = sorted(ho_sum["per_night_macro_f1"].values())
        s1_vals = sorted(is_sum["per_night_scorer1"].values())
        s2_vals = sorted(is_sum["per_night_scorer2"].values())
        external_eval = {
            "holdout": {
                **ho_sum["calibrated"],
                "conformal_90": ho_sum["conformal_calibrated"]["coverage_90"],
                "conformal_95": ho_sum["conformal_calibrated"]["coverage_95"],
                "patients": [round(v, 4) for v in ho_vals],
            },
            "isruc": {
                "scorer1": {**is_sum["calibrated_scorer1"], "patients": [round(v, 4) for v in s1_vals]},
                "scorer2": {**is_sum["calibrated_scorer2"], "patients": [round(v, 4) for v in s2_vals]},
                "scorers_agreement": lead_old["summary"]["coimbra_scorers"] if lead_old else None,
                "scorers_patients": coimbra_scorers,
                "conformal_90": is_sum["conformal_calibrated_scorer1"]["coverage_90"],
                "conformal_95": is_sum["conformal_calibrated_scorer1"]["coverage_95"],
            },
        }

    payload = {
        "generated_by": "scripts/export_concept_showcase.py",
        "active_experiment": active_exp,
        "has_external": external_eval is not None,
        "model_params": int(active_summary["model_params"]),
        "input_dim": int(active_summary["input_dim"]),
        "calibrated": {
            **active_summary["calibrated"],
            "accuracy": cv_acc,
            "subj_median": cv_subj_median,
        },
        "ladder": ladder,
        "supplementary_models": supp_models,
        "concept_attribution": {
            "local_only_macro_f1": round(float(cd["local_only_macro_f1"]), 4),
            "full_macro_f1": round(float(cd["full_macro_f1"]), 4),
            "temporal_context_gain": round(float(cd["temporal_context_gain"]), 4),
            "concepts": concepts,
            "stage_gates": gates,
        },
        "conformal": active_summary["conformal"]["calibrated"],
        "cue_screen": cues_table,
        "night_trace": night_trace,
        "external_eval": external_eval,
    }

    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"wrote {OUT.relative_to(ROOT)} (active=Exp {active_exp}, "
        f"params={payload['model_params']}, night={night_trace['cohort']}:{night_trace['record']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
