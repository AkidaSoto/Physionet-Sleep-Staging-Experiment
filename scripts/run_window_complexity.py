"""How much accuracy the window model loses as it is made simpler and easier to read.

Same window inputs as the adopted model (run_window_hmm_staging.ADOPTED, no decoder) and the same 5 folds;
only the learner changes:
  hgb_l31_i100  library-default boosting (the adopted model)
  hgb_l8_i100   8-leaf trees
  hgb_l4_i100   4-leaf trees
  hgb_l2_i200   one split per tree (boosted stumps: an additive model, one curve per input, readable like a scorecard)
  tree_d4/d6/d8 a single decision tree of that depth (rules a person can read)
Writes artifacts/staging_window_model/<dataset>/complexity/summary.csv and tree_d4.txt (the depth-4 tree's rules, fold 0).

Usage: python scripts/run_window_complexity.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import run_window_hmm_staging as run  # noqa: E402

STAGES = ["Wake", "N1", "N2", "N3", "REM"]
LEARNERS = {
    "hgb_l31_i100": lambda: HistGradientBoostingClassifier(random_state=7, class_weight="balanced"),
    "hgb_l8_i100": lambda: HistGradientBoostingClassifier(random_state=7, class_weight="balanced", max_leaf_nodes=8),
    "hgb_l4_i100": lambda: HistGradientBoostingClassifier(random_state=7, class_weight="balanced", max_leaf_nodes=4),
    "hgb_l2_i200": lambda: HistGradientBoostingClassifier(random_state=7, class_weight="balanced", max_leaf_nodes=2,
                                                          max_iter=200),
    "tree_d4": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, class_weight="balanced", random_state=7),
    "tree_d6": lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=200, class_weight="balanced", random_state=7),
    "tree_d8": lambda: DecisionTreeClassifier(max_depth=8, min_samples_leaf=200, class_weight="balanced", random_state=7),
}


def learned_values(model) -> int:
    """Split thresholds plus leaf values."""
    if isinstance(model, DecisionTreeClassifier):
        t = model.tree_
        return int(t.node_count)
    return int(sum(len(p.nodes) for it in model._predictors for p in it))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cinc2018")
    args = ap.parse_args()
    out = REPO / "artifacts/staging_window_model" / args.dataset / "complexity"
    out.mkdir(parents=True, exist_ok=True)
    subjects, rows, tracks = run.load_inputs(args.dataset, REPO / f"artifacts/{args.dataset}_window_cache")
    frame = run.epoch_frame(rows, run.WINDOW_FEATURES)
    splits = list(run.build_subject_eval_splits(groups=frame[run.SUBJ].astype(str).to_numpy(), y=frame[run.TARGET].to_numpy(),
                                                outer_validation="kfold", outer_folds=5, random_state=run.SEED))
    trk, feats = run.adopted_signals()
    names = [c.split(".")[-1] for c in trk + feats]
    dense = {s: run.dense_signals(tracks[s], rows[s], trk, feats) for s in subjects}
    y = frame[run.TARGET].to_numpy()
    preds = {name: np.zeros(len(frame), dtype=int) for name in LEARNERS}
    sizes = {name: [] for name in LEARNERS}
    for k, sp in enumerate(splits):
        tr, te = np.asarray(sp.train_idx), np.asarray(sp.test_idx)
        enc = run.WindowEncoder(names).fit(dense, sorted(set(frame.iloc[tr][run.SUBJ])))
        cols = enc.column_names()
        Xtr = pd.DataFrame(enc.transform(dense, frame.iloc[tr]).reshape(len(tr), -1), columns=cols)
        Xte = pd.DataFrame(enc.transform(dense, frame.iloc[te]).reshape(len(te), -1), columns=cols)
        for name, make in LEARNERS.items():
            model = make().fit(Xtr, y[tr])
            preds[name][te] = model.predict(Xte)
            sizes[name].append(learned_values(model))
            run.log(f"fold {k} {name}: macro-F1 {f1_score(y[te], preds[name][te], average='macro'):.4f}, {sizes[name][-1]} values")
            if name == "tree_d4" and k == 0:
                rules = export_text(model, feature_names=cols, class_names=STAGES, max_depth=4)
                (out / "tree_d4.txt").write_text(rules, encoding="utf-8")
        del Xtr, Xte
    rows_out = []
    for name in LEARNERS:
        per_fold = [f1_score(y[np.asarray(sp.test_idx)], preds[name][np.asarray(sp.test_idx)], average="macro") for sp in splits]
        stage = f1_score(y, preds[name], labels=range(5), average=None)
        rows_out.append({"learner": name, "macro_f1": float(np.mean(per_fold)), "folds": " ".join(f"{v:.3f}" for v in per_fold),
                         "kappa": float(np.mean([cohen_kappa_score(y[np.asarray(sp.test_idx)], preds[name][np.asarray(sp.test_idx)]) for sp in splits])),
                         "learned_values": int(np.mean(sizes[name])), **{f"f1_{s}": float(v) for s, v in zip(STAGES, stage)}})
        pd.DataFrame(rows_out).to_csv(out / "summary.csv", index=False)
    run.log(f"done -> {out}")
    print(pd.DataFrame(rows_out).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
