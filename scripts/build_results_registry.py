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


def _lead_figure(path: Path) -> list[Entry]:
    """Figure 1 (scripts/export_lead_figure.py): the example night and per-night score summaries."""
    data = json.loads(path.read_text(encoding="utf-8"))
    night = data["night"]
    what = f"Example held-out night {night['record']} (closest to the held-out median)"
    out = [("lead.night.macro_f1", night["macro_f1"], f"{what}, macro-F1 over the stages present"),
           ("lead.night.accuracy", night["accuracy"], f"{what}, accuracy"),
           ("lead.night.epochs", float(night["epochs"]), f"{what}, scored epochs"),
           ("lead.night.disagreements", float(night["disagreements"]), f"{what}, epochs where model and expert differ")]
    for cohort, s in data["summary"].items():
        out += [(f"lead.{cohort}.{k}", float(s[k]), f"Per-night macro-F1 ({data['definition']}) {cohort}, {k}")
                for k in ("n", "median", "min", "max", "q1", "q3")]
    return out


def _development_comparison(path: Path) -> list[Entry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    variant = data["variant"]
    prefix = f"dev.waveform.{variant}"
    entries = [(f"{prefix}.{key}", data["candidate"][key], f"{variant}, Boston development CV {key}")
               for key in ("macro_f1", "f1_N1", "f1_N2")]
    entries.append((f"{prefix}.delta", data["macro_f1_delta"], f"{variant} minus Model C, paired macro-F1"))
    return entries


def _subject_budget_comparison(path: Path) -> list[Entry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    small = data["train_128"]["macro_f1"]
    full = data["full_budget"]["macro_f1"]
    return [
        ("dev.subject_budget.train128.macro_f1", small, "Model C on 128 training patients per fold"),
        ("dev.subject_budget.full.macro_f1", full, "Model C on the full training budget per fold"),
        ("dev.subject_budget.delta", full - small, "Full minus 128 training patients, same evaluation folds"),
    ]


def _waveform_probability_blend(path: Path) -> list[Entry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    blend = data["blend"]["macro_f1"]
    return [
        ("dev.waveform.blend.macro_f1", blend, "Fixed compact HGB and ResNet blend, Boston development CV"),
        ("dev.waveform.blend.delta", blend - data["hgb18"]["macro_f1"], "Fixed blend minus compact HGB, same evaluation folds"),
    ]


def _tuned_features(variant: str) -> Callable[[Path], list[Entry]]:
    def read(path: Path) -> list[Entry]:
        summary = json.loads(path.read_text(encoding="utf-8"))
        data = summary[variant]
        prefix = f"dev.features.{variant}"
        entries = [
            (f"{prefix}.macro_f1", data["candidate"]["macro_f1"], f"{variant} feature recipe, Boston development CV"),
            (f"{prefix}.delta", data["delta"], f"{variant} feature recipe versus the archived 18-input baseline"),
        ]
        if variant == "reduced":
            entries.append((f"{prefix}.delta_vs_joint", summary["reduced_vs_joint"]["delta"],
                            "16-input recipe minus the selected 18-input recipe"))
        return entries

    return read


def _feature_context(variants: tuple[str, ...]) -> Callable[[Path], list[Entry]]:
    def read(path: Path) -> list[Entry]:
        results = json.loads(path.read_text(encoding="utf-8"))["contrasts"]
        entries = []
        for variant in variants:
            data = results[f"{variant}_vs_baseline"]
            label = f"Compact feature context, {variant}"
            entries.extend([
                (f"dev.context.{variant}.macro_f1", data["candidate"]["macro_f1"], f"{label}, Boston development CV"),
                (f"dev.context.{variant}.delta", data["delta"], f"{label} versus compact HGB"),
            ])
        reference = "short_blend" if variants[0].endswith("blend") else "short"
        candidate = "long_blend" if reference.endswith("blend") else "long"
        contrast = f"{candidate}_vs_{reference}"
        entries.append((f"dev.context.{contrast}", results[contrast]["delta"], "Long context minus matched short context"))
        return entries

    return read


def _probability_diagnostic(variants: tuple[str, ...]) -> Callable[[Path], list[Entry]]:
    def read(path: Path) -> list[Entry]:
        metrics = json.loads(path.read_text(encoding="utf-8"))["probability"]
        entries = []
        for variant in variants:
            result = metrics[variant]
            for key in ("log_loss", "brier", "macro_f1"):
                value = (sum(f[key] for f in result["folds"]) / len(result["folds"])
                         if key == "macro_f1" else result["pooled"][key])
                entries.append((f"dev.probability.{variant}.{key}", value, f"Boston probability diagnostic, {variant}, {key}"))
        return entries

    return read


def _recipe_search(path: Path) -> list[Entry]:
    metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    entries = []
    for variant in ("searched", "fixed_context"):
        for key in ("macro_f1", "delta", "log_loss", "brier"):
            entries.append((f"dev.recipe_search.{variant}.{key}", metrics[variant][key],
                            f"Boston recipe and context search, {variant}, {key}"))
    return entries


def _glassbox(path: Path) -> list[Entry]:
    report = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for variant in ("additive", "pairwise"):
        comparison = report["contrasts"][f"{variant}_vs_baseline"]
        entries.extend([
            (f"dev.glassbox.{variant}.macro_f1", comparison["candidate"]["macro_f1"],
             f"Boston explainable boosting, {variant}, mean fold macro-F1"),
            (f"dev.glassbox.{variant}.log_loss", report["probabilities"][variant]["log_loss"],
             f"Boston explainable boosting, {variant}, pooled log loss"),
        ])
    entries.append(("dev.glassbox.interaction_delta", report["contrasts"]["pairwise_vs_additive"]["delta"],
                    "Boston explainable boosting, macro-F1 change from adding the seven pair terms"))
    return entries


def _sustained_errors(path: Path) -> list[Entry]:
    scope = json.loads(path.read_text(encoding="utf-8"))
    return [
        ("dev.sustained.epochs", scope["same_rival_sustained_epochs"],
         "Boston development epochs in sustained same-rival errors of both trees"),
        ("dev.sustained.bouts", scope["same_rival_sustained_bouts"],
         "Boston development runs of sustained same-rival errors of both trees"),
    ]


def _gated_temporal(variants: tuple[str, ...], reference: str) -> Callable[[Path], list[Entry]]:
    def read(path: Path) -> list[Entry]:
        study = json.loads(path.read_text(encoding="utf-8"))
        entries = []
        for model in variants:
            row = study["contrasts"][f"{model}_vs_baseline"]["candidate"]
            entries.append((f"dev.gated_temporal.{model}.macro_f1", row["macro_f1"],
                            f"Boston directed temporal model {model}, mean fold macro-F1"))
        model = variants[-1]
        delta = study["contrasts"][f"{model}_vs_{reference}"]["delta"]
        entries.append((f"dev.gated_temporal.{model}.delta", delta,
                        f"Boston directed temporal model {model}, paired macro-F1 change versus {reference}"))
        return entries

    return read


def _complementary_probes(path: Path) -> list[Entry]:
    study = json.loads(path.read_text(encoding="utf-8"))
    return [
        (f"dev.complementary.{pair.lower()}.{variant}.hard_auc", result["hard"]["auc"],
         f"Boston complementary marker probe, {pair}, {variant}, hard-case AUC")
        for pair, comparisons in study.items() for variant, result in comparisons.items()
    ]


def _complementary_model(path: Path) -> list[Entry]:
    study = json.loads(path.read_text(encoding="utf-8"))
    match = study["contrasts"]["expanded_vs_gated"]
    return [
        ("dev.complementary.expanded.macro_f1", match["candidate"]["macro_f1"],
         "Boston complementary-marker gated network, mean fold macro-F1"),
        ("dev.complementary.expanded.delta", match["delta"],
         "Boston complementary-marker gated network, matched macro-F1 change"),
    ]


def _targeted_refinement(path: Path) -> list[Entry]:
    study = json.loads(path.read_text(encoding="utf-8"))
    return [
        (f"dev.targeted.{variant}.{key}", score[key],
         f"Boston targeted refinement agreement audit, {variant}, mean fold {key}")
        for variant, score in study["agreement"].items()
        for key in ("kappa", "macro_f1")
    ]


def _stage_pair_head(head: str) -> Callable[[Path], list[Entry]]:
    def read(path: Path) -> list[Entry]:
        study = json.loads(path.read_text(encoding="utf-8"))
        comparison = study["contrasts"][head]["against_expanded"]
        scores = comparison["candidate"]
        n1 = study["metrics"][head]["stages"]["N1"]
        return [
            (f"dev.stage_pair.{head}.macro_f1", scores["macro_f1"], f"Boston {head}, mean fold macro-F1"),
            (f"dev.stage_pair.{head}.kappa", scores["kappa"], f"Boston {head}, mean fold kappa"),
            (f"dev.stage_pair.{head}.n1_recall", n1["recall"], f"Boston {head}, pooled N1 recall"),
        ]

    return read


def _compact_markers(path: Path) -> list[Entry]:
    study = json.loads(path.read_text(encoding="utf-8"))
    scores = study["contrasts"]["reduced_vs_expanded"]["candidate"]
    return [
        ("dev.stage_pair.compact.macro_f1", scores["macro_f1"], "Boston 21-measurement refit, mean fold macro-F1"),
        ("dev.stage_pair.compact.kappa", scores["kappa"], "Boston 21-measurement refit, mean fold kappa"),
    ]


def _current_data_curve(path: Path) -> list[Entry]:
    study = json.loads(path.read_text(encoding="utf-8"))
    comparison = study["contrasts"]["full_vs_train_128"]
    return [
        (f"dev.current_curve.{arm}.{metric}", comparison[role][metric],
         f"Boston expanded model, {arm} fitting budget, {metric}")
        for arm, role in (("train_128", "reference"), ("full", "candidate"))
        for metric in ("macro_f1", "kappa", "f1_N1")
    ]


def _wake_n1_timing(path: Path) -> list[Entry]:
    study = json.loads(path.read_text(encoding="utf-8"))
    return [
        (f"dev.wake_n1.timing.{variant}.auc", auc, f"Boston hard Wake/N1 {variant}, mean fold AUC")
        for variant, auc in study["mean_auc"].items()
    ]


def _wake_n1_followup(path: Path) -> list[Entry]:
    study = json.loads(path.read_text(encoding="utf-8"))
    return [
        (f"dev.wake_n1.followup.{variant}.{metric}", score,
         f"Boston Wake/N1 refinement {variant}, {metric}")
        for variant, comparison in study["contrasts"].items()
        for metric, score in (
            ("macro_f1_delta", comparison["delta"]),
            ("target_error_reduction", comparison["target_error_reduction"]),
            ("n1_f1_delta", comparison["stage_deltas"]["N1"]),
        )
    ]


def _additional_training(path: Path) -> list[Entry]:
    study = json.loads(path.read_text(encoding="utf-8"))
    comparison = study["comparison"]
    return [
        (f"dev.additional_training.{arm}.{metric}", comparison[arm][metric],
         f"Boston fixed folds, {arm} fitting pool, {metric}")
        for arm in ("reference", "candidate")
        for metric in ("macro_f1", "kappa", "f1_Wake", "f1_N1")
    ] + [
        (f"dev.additional_training.{name}", comparison[name], f"Boston training expansion, {name}")
        for name in ("delta", "wins", "kappa_delta")
    ] + [
        (f"dev.additional_training.{arm}.{name}", value, f"Boston training expansion, {arm}, {name}")
        for arm in ("original", "expanded")
        for name, value in (
            ("hard_wake_n1_auc", study["wake_n1"][arm]["auc_hard"]),
            ("local_hard_wake_n1_auc", study["wake_n1_mechanisms"][arm]["hard_local_auc"]),
        )
    ]


def _bigru75(path: Path) -> list[Entry]:
    study = json.loads(path.read_text(encoding="utf-8"))
    out: list[Entry] = []
    for arm in ("uncalibrated", "calibrated"):
        s = study[arm]
        key = "raw" if arm == "uncalibrated" else "calibrated"
        out += [
            (f"dev.bigru75.{key}.macro_f1", float(s["mean_fold_macro_f1"]), f"#75 Unified BiGRU ({key}), macro-F1"),
            (f"dev.bigru75.{key}.kappa", float(s["kappa"]), f"#75 Unified BiGRU ({key}), kappa"),
            (f"dev.bigru75.{key}.log_loss", float(s["log_loss"]), f"#75 Unified BiGRU ({key}), log loss"),
        ]
        for st in STAGES:
            out.append((f"dev.bigru75.{key}.f1_{st}", float(s["stage_f1"][st]), f"#75 Unified BiGRU ({key}), {st} F1"))
    cmp73 = study["reference_73_comparison"]["calibrated_vs_73"]
    out.append(("dev.bigru75.delta_vs_73", float(cmp73["macro_f1_delta"]), "#75 vs #73 calibrated macro-F1 delta"))
    return out


def _hgb704(path: Path) -> list[Entry]:
    study = json.loads(path.read_text(encoding="utf-8"))
    out: list[Entry] = [
        ("dev.hgb704.n_features", float(study["n_features"]), "#76 Multi-scale HGB (704 pts), feature count"),
    ]
    for arm in ("raw", "calibrated"):
        s = study[arm]
        out += [
            (f"dev.hgb704.{arm}.macro_f1", float(s["macro_f1"]), f"#76 HGB-704 ({arm}), macro-F1"),
            (f"dev.hgb704.{arm}.kappa", float(s["kappa"]), f"#76 HGB-704 ({arm}), kappa"),
        ]
        for st in STAGES:
            out.append((f"dev.hgb704.{arm}.f1_{st}", float(s[f"f1_{st}"]), f"#76 HGB-704 ({arm}), {st} F1"))
    return out


def _ensemble78(path: Path) -> list[Entry]:
    study = json.loads(path.read_text(encoding="utf-8"))
    best = study["ensembles"][study["best_ensemble"]]
    out: list[Entry] = [
        ("dev.ensemble78.best.macro_f1", float(best["macro_f1"]), "#78 Unified ensemble, macro-F1"),
        ("dev.ensemble78.best.kappa", float(best["kappa"]), "#78 Unified ensemble, kappa"),
        ("dev.ensemble78.best.accuracy", float(best["accuracy"]), "#78 Unified ensemble, accuracy"),
        ("dev.ensemble78.best.log_loss", float(best["log_loss"]), "#78 Unified ensemble, log loss"),
    ]
    for st in STAGES:
        out.append((f"dev.ensemble78.best.f1_{st}", float(best["stage_f1"][st]), f"#78 Unified ensemble, {st} F1"))
    return out


def _concept_model(prefix: str, label: str) -> Callable[[Path], list[Entry]]:
    def read(path: Path) -> list[Entry]:
        study = json.loads(path.read_text(encoding="utf-8"))
        out: list[Entry] = [
            (f"{prefix}.params", float(study["model_params"]), f"{label}, trainable parameters"),
            (f"{prefix}.input_dim", float(study["input_dim"]), f"{label}, input dimension"),
        ]
        for arm in ("raw", "calibrated"):
            if arm not in study:
                continue
            s = study[arm]
            out += [
                (f"{prefix}.{arm}.macro_f1", float(s["mean_fold_macro_f1"]), f"{label} ({arm}), macro-F1"),
                (f"{prefix}.{arm}.pooled_macro_f1", float(s["pooled_macro_f1"]), f"{label} ({arm}), pooled macro-F1"),
                (f"{prefix}.{arm}.kappa", float(s["kappa"]), f"{label} ({arm}), kappa"),
                (f"{prefix}.{arm}.log_loss", float(s["log_loss"]), f"{label} ({arm}), log loss"),
                (f"{prefix}.{arm}.brier", float(s["brier"]), f"{label} ({arm}), Brier score"),
            ]
            for st in STAGES:
                out.append((f"{prefix}.{arm}.f1_{st}", float(s["stage_f1"][st]), f"{label} ({arm}), {st} F1"))
        cd = study.get("concept_diagnostics", {})
        if cd:
            out += [
                (f"{prefix}.local_only_macro_f1", float(cd["local_only_macro_f1"]), f"{label}, local-only macro-F1"),
                (f"{prefix}.temporal_gain", float(cd["temporal_context_gain"]), f"{label}, temporal context gain"),
            ]
            for c_name, c_vals in cd.get("concept_ablation", {}).items():
                out.append((f"{prefix}.ablation.{c_name}_local_drop", float(c_vals["local_macro_f1_drop"]), f"{label}, local drop removing {c_name}"))
                out.append((f"{prefix}.ablation.{c_name}_full_drop", float(c_vals["full_direct_drop"]), f"{label}, full drop removing {c_name}"))
            for st, g_vals in cd.get("stage_gates", {}).items():
                out.append((f"{prefix}.gate.{st}.past", float(g_vals["past_gate_mean"]), f"{label}, mean {st} past gate"))
                out.append((f"{prefix}.gate.{st}.future", float(g_vals["future_gate_mean"]), f"{label}, mean {st} future gate"))
        conf_cal = study.get("conformal", {}).get("calibrated", {})
        for level in ("coverage_90", "coverage_95"):
            if level not in conf_cal:
                continue
            for mode in ("marginal", "mondrian"):
                m = conf_cal[level][mode]
                out += [
                    (f"{prefix}.conformal.{level}.{mode}.coverage", float(m["empirical_coverage"]), f"{label} {level} {mode}, coverage"),
                    (f"{prefix}.conformal.{level}.{mode}.set_size", float(m["mean_set_size"]), f"{label} {level} {mode}, mean set size"),
                    (f"{prefix}.conformal.{level}.{mode}.singleton_rate", float(m["singleton_rate"]), f"{label} {level} {mode}, singleton rate"),
                    (f"{prefix}.conformal.{level}.{mode}.singleton_acc", float(m["singleton_accuracy"]), f"{label} {level} {mode}, singleton accuracy"),
                    (f"{prefix}.conformal.{level}.{mode}.n1_coverage", float(m["per_stage"]["N1"]["coverage"]), f"{label} {level} {mode}, N1 coverage"),
                ]
                if "transition_epochs" in m:
                    out.append((f"{prefix}.conformal.{level}.{mode}.transition_size", float(m["transition_epochs"]["mean_set_size"]), f"{label} {level} {mode}, transition mean set size"))
                    out.append((f"{prefix}.conformal.{level}.{mode}.interior_size", float(m["interior_epochs"]["mean_set_size"]), f"{label} {level} {mode}, interior mean set size"))
        cmps = study.get("comparisons", {})
        for cmp_name, cmp_obj in cmps.items():
            if isinstance(cmp_obj, dict) and "calibrated_macro_f1_delta" in cmp_obj:
                d = cmp_obj["calibrated_macro_f1_delta"]
                out.append((f"{prefix}.{cmp_name}.delta", float(d["mean"]), f"{label}, calibrated delta {cmp_name}"))
                out.append((f"{prefix}.{cmp_name}.ci_lo", float(d["ci95"][0]), f"{label}, calibrated delta {cmp_name} CI low"))
                out.append((f"{prefix}.{cmp_name}.ci_hi", float(d["ci95"][1]), f"{label}, calibrated delta {cmp_name} CI high"))
        for key, short in (("vs_79_calibrated_delta", "vs_79"), ("vs_75_calibrated_delta", "vs_75_calibrated")):
            if isinstance(study.get(key), dict) and "mean" in study[key]:
                d = study[key]
                out.append((f"{prefix}.{short}.delta", float(d["mean"]), f"{label}, calibrated delta {short}"))
                out.append((f"{prefix}.{short}.ci_lo", float(d["ci95"][0]), f"{label}, calibrated delta {short} CI low"))
                out.append((f"{prefix}.{short}.ci_hi", float(d["ci95"][1]), f"{label}, calibrated delta {short} CI high"))
        return out
    return read


def _cohen_d_screen(path: Path) -> list[Entry]:
    screen = json.loads(path.read_text(encoding="utf-8"))
    out: list[Entry] = []
    for cue, vals in screen.items():
        out.append((f"dev.cue_screen.{cue}.max_abs_d", float(vals["max_abs_d"]), f"Training-fold Cohen's d max |d| for {cue}"))
        for c_name, d_val in vals["mean_cohen_d"].items():
            out.append((f"dev.cue_screen.{cue}.{c_name}", float(d_val), f"Training-fold Cohen's d {cue} on {c_name}"))
    return out


def _concept_external_holdout(prefix: str, label: str) -> Callable[[Path], list[Entry]]:
    def read(path: Path) -> list[Entry]:
        study = json.loads(path.read_text(encoding="utf-8"))
        out: list[Entry] = []
        for arm in ("calibrated", "raw"):
            s = study[arm]
            out += [
                (f"{prefix}.holdout.{arm}.macro_f1", float(s["macro_f1"]), f"{label} Boston holdout ({arm}), macro-F1"),
                (f"{prefix}.holdout.{arm}.kappa", float(s["kappa"]), f"{label} Boston holdout ({arm}), kappa"),
                (f"{prefix}.holdout.{arm}.accuracy", float(s["accuracy"]), f"{label} Boston holdout ({arm}), accuracy"),
                (f"{prefix}.holdout.{arm}.log_loss", float(s["log_loss"]), f"{label} Boston holdout ({arm}), log loss"),
                (f"{prefix}.holdout.{arm}.subj_median", float(s["subj_median"]), f"{label} Boston holdout ({arm}), subject median"),
            ]
            for st in STAGES:
                out.append((f"{prefix}.holdout.{arm}.f1_{st}", float(s["stage_f1"][st]), f"{label} Boston holdout ({arm}), {st} F1"))
        for cov in ("coverage_90", "coverage_95"):
            c = study["conformal_calibrated"][cov]
            out += [
                (f"{prefix}.holdout.conformal.{cov}.coverage", float(c["empirical_coverage"]), f"{label} Boston holdout {cov}, empirical coverage"),
                (f"{prefix}.holdout.conformal.{cov}.set_size", float(c["mean_set_size"]), f"{label} Boston holdout {cov}, mean set size"),
                (f"{prefix}.holdout.conformal.{cov}.singleton_rate", float(c["singleton_rate"]), f"{label} Boston holdout {cov}, singleton rate"),
                (f"{prefix}.holdout.conformal.{cov}.singleton_acc", float(c["singleton_accuracy"]), f"{label} Boston holdout {cov}, singleton accuracy"),
                (f"{prefix}.holdout.conformal.{cov}.n1_coverage", float(c["per_stage_coverage"]["N1"]), f"{label} Boston holdout {cov}, N1 coverage"),
                (f"{prefix}.holdout.conformal.{cov}.transition_size", float(c["transition_set_size"]), f"{label} Boston holdout {cov}, transition set size"),
                (f"{prefix}.holdout.conformal.{cov}.interior_size", float(c["interior_set_size"]), f"{label} Boston holdout {cov}, interior set size"),
            ]
        return out
    return read


def _concept_external_isruc(prefix: str, label: str) -> Callable[[Path], list[Entry]]:
    def read(path: Path) -> list[Entry]:
        study = json.loads(path.read_text(encoding="utf-8"))
        out: list[Entry] = []
        for arm in ("calibrated_scorer1", "calibrated_scorer2", "raw_scorer1", "raw_scorer2"):
            s = study[arm]
            out += [
                (f"{prefix}.isruc.{arm}.macro_f1", float(s["macro_f1"]), f"{label} Coimbra ISRUC {arm}, macro-F1"),
                (f"{prefix}.isruc.{arm}.kappa", float(s["kappa"]), f"{label} Coimbra ISRUC {arm}, kappa"),
                (f"{prefix}.isruc.{arm}.accuracy", float(s["accuracy"]), f"{label} Coimbra ISRUC {arm}, accuracy"),
                (f"{prefix}.isruc.{arm}.subj_median", float(s["subj_median"]), f"{label} Coimbra ISRUC {arm}, subject median"),
            ]
            for st in STAGES:
                out.append((f"{prefix}.isruc.{arm}.f1_{st}", float(s["stage_f1"][st]), f"{label} Coimbra ISRUC {arm}, {st} F1"))
        for cov in ("coverage_90", "coverage_95"):
            c = study["conformal_calibrated_scorer1"][cov]
            out += [
                (f"{prefix}.isruc.conformal.{cov}.coverage", float(c["empirical_coverage"]), f"{label} Coimbra ISRUC {cov}, empirical coverage"),
                (f"{prefix}.isruc.conformal.{cov}.set_size", float(c["mean_set_size"]), f"{label} Coimbra ISRUC {cov}, mean set size"),
                (f"{prefix}.isruc.conformal.{cov}.singleton_acc", float(c["singleton_accuracy"]), f"{label} Coimbra ISRUC {cov}, singleton accuracy"),
            ]
        return out
    return read


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
    Source("showcase/content/generated/lead-figure.json", _lead_figure, exp=37),
]
MEASURED += [
    source for source in (
        Source("artifacts/exploration/cinc-reduction18/reduction/summary.json", _development_comparison, exp=39),
        Source("artifacts/exploration/cinc-waveform/scratch/summary.json", _development_comparison, exp=40),
        Source("artifacts/exploration/cinc-cap-audit/learning_curve_summary.json", _subject_budget_comparison, exp=41),
        Source("artifacts/exploration/cinc-cap-audit/fusion_summary.json", _waveform_probability_blend, exp=42),
        Source("artifacts/exploration/cinc-feature-tuning/summary.json", _tuned_features("joint"), exp=45),
        Source("artifacts/exploration/cinc-feature-tuning/summary.json", _tuned_features("reduced"), exp=46),
        Source("artifacts/exploration/cinc-feature-context/summary.json", _feature_context(("short", "long")), exp=47),
        Source("artifacts/exploration/cinc-feature-context/summary.json", _feature_context(("short_blend", "long_blend")), exp=48),
        Source("artifacts/exploration/cinc-probability-diagnostic/summary.json", _probability_diagnostic(("tree", "sequence", "calibrated", "tree_prior")), exp=49),
        Source("artifacts/exploration/cinc-probability-diagnostic/summary.json", _probability_diagnostic(("hard", "soft")), exp=50),
        Source("artifacts/exploration/cinc-probability-diagnostic/summary.json", _probability_diagnostic(("clock", "physiology")), exp=51),
        Source("artifacts/exploration/cinc-recipe-search/summary.json", _recipe_search, exp=53),
        Source("artifacts/exploration/cinc-error-interactions/glassbox_summary.json", _glassbox, exp=56),
        Source("artifacts/exploration/cinc-gated-temporal/sustained_scope.json", _sustained_errors, exp=57),
        Source("artifacts/exploration/cinc-gated-temporal/summary.json", _gated_temporal(("ungated", "gated"), "ungated"), exp=58),
        Source("artifacts/exploration/cinc-gated-temporal/summary.json", _gated_temporal(("gated_pair",), "gated"), exp=59),
        Source("artifacts/exploration/cinc-complementary-markers/probe_summary.json", _complementary_probes, exp=60),
        Source("artifacts/exploration/cinc-complementary-markers/model_summary.json", _complementary_model, exp=61),
        Source("artifacts/exploration/cinc-targeted-refinement/summary.json", _targeted_refinement, exp=62),
        Source("artifacts/exploration/cinc-stage-pair-revisit/head_summary.json", _stage_pair_head("head"), exp=65),
        Source("artifacts/exploration/cinc-stage-pair-revisit/head_summary.json", _stage_pair_head("k_head"), exp=66),
        Source("artifacts/exploration/cinc-stage-pair-revisit/head_summary.json", _stage_pair_head("continuation_head"), exp=67),
        Source("artifacts/exploration/cinc-stage-pair-revisit/reduction/summary.json", _compact_markers, exp=68),
        Source("artifacts/exploration/cinc-wake-n1-data/data_curve/summary.json", _current_data_curve, exp=70),
        Source("artifacts/exploration/cinc-wake-n1-followup/probe_summary.json", _wake_n1_timing, exp=71),
        Source("artifacts/exploration/cinc-wake-n1-followup/model_summary.json", _wake_n1_followup, exp=72),
        Source("artifacts/exploration/cinc-all-additional-training/summary.json", _additional_training, exp=73),
        Source("artifacts/exploration/cinc-bigru-unified/summary.json", _bigru75, exp=75),
        Source("artifacts/exploration/cinc-hgb-704/summary.json", _hgb704, exp=76),
        Source("artifacts/exploration/cinc-unified-ensemble/summary.json", _ensemble78, exp=78),
        Source("artifacts/exploration/cinc-bigru-concept/summary.json", _concept_model("dev.concept79", "#79 Concept-BiGRU"), exp=79),
        Source("artifacts/exploration/cinc-bigru-concept/external_eval/holdout_summary.json", _concept_external_holdout("ext.concept79", "#79"), exp=81),
        Source("artifacts/exploration/cinc-bigru-concept/external_eval/isruc_summary.json", _concept_external_isruc("ext.concept79", "#79"), exp=81),
        Source("artifacts/exploration/cinc-bigru-cues/cohen_d_screen.json", _cohen_d_screen, exp=80),
        Source("artifacts/exploration/cinc-bigru-cues/rem_auto/summary.json", _concept_model("dev.cues80", "#80 Concept-BiGRU + Cues"), exp=80),
        Source("artifacts/exploration/cinc-bigru-pruned/minimal13/summary.json", _concept_model("dev.minimal82", "#82 Concept-BiGRU minimal13"), exp=82),
        Source("artifacts/exploration/cinc-bigru-pruned/minimal13/external_eval/holdout_summary.json", _concept_external_holdout("ext.minimal82", "#82 minimal13"), exp=82),
        Source("artifacts/exploration/cinc-bigru-pruned/minimal13/external_eval/isruc_summary.json", _concept_external_isruc("ext.minimal82", "#82 minimal13"), exp=82),
    ) if (ROOT / source.path).is_file()
]


def derive(v: dict[str, float]) -> dict[str, tuple[float, list[str], str]]:
    """Derived entries: id -> (value, input ids, label). Pure arithmetic on measured values."""
    def diff(a: str, b: str, label: str):
        return v[a] - v[b], [a, b], label
    spindle = ["cv.nocardio27_bursts.epoch.macro_f1", "cv.nocardio27.epoch.macro_f1",
               "cv.prune6_sigratio.epoch.macro_f1", "cv.prune6.epoch.macro_f1", "cv.prune6_bursts.epoch.macro_f1"]
    cardio = [f"cv.{c}.{d}.macro_f1" for c in ("cardio6", "cardio6_c5") for d in ("epoch", "hmm")]
    out = {
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
        "delta.concept79_holdout_vs_model_c": diff("ext.concept79.holdout.calibrated.macro_f1", "holdout.look2.g20.macro_f1",
                                                  "#79 Concept-BiGRU minus Model C on 200 held-out Boston patients"),
        "delta.concept79_isruc_s1_vs_model_c": diff("ext.concept79.isruc.calibrated_scorer1.macro_f1", "coimbra.transfer_cinc_to_isruc_scorer1.macro_f1",
                                                   "#79 Concept-BiGRU minus Model C on Coimbra ISRUC scorer 1"),
        "delta.concept79_isruc_s2_vs_model_c": diff("ext.concept79.isruc.calibrated_scorer2.macro_f1", "coimbra.transfer_cinc_to_isruc_scorer2.macro_f1",
                                                   "#79 Concept-BiGRU minus Model C on Coimbra ISRUC scorer 2"),
    }
    if "ext.minimal82.holdout.calibrated.macro_f1" in v:
        out["delta.minimal82_holdout_vs_model_c"] = diff(
            "ext.minimal82.holdout.calibrated.macro_f1", "holdout.look2.g20.macro_f1",
            "#82 Concept-BiGRU minimal13 minus Model C on 200 held-out Boston patients",
        )
    if "ext.minimal82.isruc.calibrated_scorer1.macro_f1" in v:
        out["delta.minimal82_isruc_s1_vs_model_c"] = diff(
            "ext.minimal82.isruc.calibrated_scorer1.macro_f1", "coimbra.transfer_cinc_to_isruc_scorer1.macro_f1",
            "#82 Concept-BiGRU minimal13 minus Model C on Coimbra ISRUC scorer 1",
        )
        out["delta.minimal82_isruc_s2_vs_model_c"] = diff(
            "ext.minimal82.isruc.calibrated_scorer2.macro_f1", "coimbra.transfer_cinc_to_isruc_scorer2.macro_f1",
            "#82 Concept-BiGRU minimal13 minus Model C on Coimbra ISRUC scorer 2",
        )
    return out


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
