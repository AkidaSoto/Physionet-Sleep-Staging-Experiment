"""Fit the adopted window model on fold 0's training subjects and count its learned values; count CNN weights."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path.cwd(); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))  # run from the repo root
import run_window_hmm_staging as r
from physionet_sleep.experiments.window_cnn import EpochNet
subjects, rows, tracks = r.load_inputs("cinc2018", REPO / "artifacts/cinc2018_window_cache")
frame = r.epoch_frame(rows, r.WINDOW_FEATURES)
sp = list(r.build_subject_eval_splits(groups=frame[r.SUBJ].astype(str).to_numpy(), y=frame[r.TARGET].to_numpy(), outer_validation="kfold", outer_folds=5, random_state=7))[0]
train = frame.iloc[sp.train_idx]
trk, feats = r.adopted_signals()
names = trk + feats
dense = {s: r.dense_signals(tracks[s], rows[s], trk, feats) for s in subjects}
enc = r.WindowEncoder([c.split(".")[-1] for c in names]).fit(dense, sorted(set(train[r.SUBJ])))
W = enc.transform(dense, train)
t = time.time()
model, _ = r.fit_proba(W.reshape(len(W), -1), train[r.TARGET].to_numpy(), W[:5].reshape(5, -1), enc.column_names())
fit_s = time.time() - t
est = model[-1] if hasattr(model, "__getitem__") else model
preds = est._predictors
n_trees = sum(len(it) for it in preds)
nodes = sum(len(p.nodes) for it in preds for p in it)
leaves = sum(int(p.nodes["is_leaf"].sum()) for it in preds for p in it)
used = len({int(n["feature_idx"]) for it in preds for p in it for n in p.nodes if not n["is_leaf"]})
cnn = sum(p.numel() for p in EpochNet(30).parameters())
out = dict(n_iter=int(est.n_iter_), trees=n_trees, nodes=nodes, leaves=leaves, splits=nodes - leaves, inputs_used=used,
           inputs=W.shape[1] * W.shape[2], train_subjects=int(train[r.TARGET].size and train[r.SUBJ].nunique()),
           train_epochs=len(train), fit_seconds=round(fit_s), hmm_params=25, cnn_weights=cnn)
print(json.dumps(out))
(REPO / "artifacts/staging_window_model/cinc2018/model_size.json").write_text(json.dumps(out, indent=2))
