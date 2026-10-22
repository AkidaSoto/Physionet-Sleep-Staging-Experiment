"""Whole-night sequence models for no-EEG staging (heart + breathing only, CinC 2018 development subjects).

Each night is one sequence of per-epoch vectors (every epoch of the recording; unlabelled epochs are inputs only).
Inputs (src/physionet_sleep/experiments/noeeg.py), z-scored with training-fold statistics, clipped to +-6, missing -> 0:
  base   15 per-epoch summaries of the 3 heart/breathing tracks and 9 features
  fine   + 6-s step means of the 3 tracks within the epoch (15)
  raw    + 25 raw-derived summaries (R-peak HRV, breath-by-breath, movement, SpO2); rawnospo2 without the 4 SpO2;
         or groups: rawheart (10), rawbreath (8), rawmove (3), rawspo2 (4)
  ecgonly / beltonly   one sensor: its base, 6-s and raw columns (ECG: HR/HRV + ECG noise; belts: breathing + belt
         movement); add spo2 for the oximeter, e.g. tcn_ecgonly_spo2_pos
  ihr    + a small 1-D CNN over the epoch's 4-Hz instantaneous heart rate (120 samples + 30 s each side)
  pos    + night position (3)
Models: bidirectional GRU (2 x 64) or dilated TCN (64 channels, dilations 1..128, non-causal). Class-balanced
cross-entropy, Adam, early stopping on macro-F1 of 10 % of the training subjects; probabilities averaged over seeds;
reported epoch-only and with the same HMM decoder as the tree models.

Usage: python scripts/run_noeeg_sequence.py --variant gru_base_pos [--seeds 0,1] [--folds 0,1,2,3,4] [--batch 16] [--hidden 64]
  variant = <gru|tcn>_<inputs joined by _>, e.g. gru_base_pos, tcn_base_fine_raw_pos, gru_base_fine_raw_ihr_pos
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from physionet_sleep.analysis.epoch_windows import SUBJ, TARGET  # noqa: E402
from physionet_sleep.experiments import noeeg as ne  # noqa: E402

torch.set_num_threads(4)
OUT = ne.NOEEG / "sequence"
IHR_CTX = 30  # seconds of heart rate on each side of the epoch for the CNN front-end


# ---------------------------------------------------------------- data
BASE_NAMES = ["hr_mean", "resp_effort_mean", "resp_env_mean", "hr_sd", "resp_effort_sd", "resp_env_sd"] + [
    c.split(".")[-1] for c in ne.CARDIO_FULL]
SENSOR = {  # single-sensor inputs: (base columns, fine track index, raw columns)
    "ecgonly": (["hr_mean", "hr_sd", "hr_bpm_30s", "hr_surge_30s", "sdnn_150s", "hr_slow_sd_300s", "dfa_a2_300s",
                 "ln_lf_hf_300s"], [0], ne.RAW_GROUPS["heart"] + ["ecg_noise"]),
    "beltonly": (["resp_effort_mean", "resp_env_mean", "resp_effort_sd", "resp_env_sd", "breath_interval_cv_60s",
                  "breath_amp_cv_60s"], [1, 2], ne.RAW_GROUPS["breath"] + ["effort_move", "effort_hf"]),
}


def night_inputs(s: str, parts: set[str], blocks: dict) -> tuple[np.ndarray, np.ndarray | None]:
    b = blocks[s]
    nep = len(b["base"])
    sensor = next((k for k in SENSOR if k in parts), None)
    if sensor:  # one sensor only (+ SpO2 with "spo2"), base + fine + its raw columns
        bc, tracks_i, rc = SENSOR[sensor]
        rc = rc + (ne.SPO2_COLS if "spo2" in parts else [])
        X = [b["base"][:, [BASE_NAMES.index(c) for c in bc]],
             b["fine"][:, [3 * k + t for k in range(5) for t in tracks_i]],
             b["raw"][:, [ne.RAW_COLS.index(c) for c in rc]]]
    else:
        X = [b[k] for k in ("base", "fine") if k in parts]
    if sensor:
        pass  # raw columns already chosen
    elif "raw" in parts:
        X.append(b["raw"])
    elif "rawnospo2" in parts:
        X.append(b["raw"][:, [i for i, c in enumerate(ne.RAW_COLS) if c not in ne.SPO2_COLS]])
    else:  # raw groups one by one, e.g. rawheart, rawbreath, rawmove, rawspo2
        cols = [c for g, cs in ne.RAW_GROUPS.items() if f"raw{g}" in parts for c in cs]
        if cols:
            X.append(b["raw"][:, [ne.RAW_COLS.index(c) for c in cols]])
    if "pos" in parts:
        X.append(b["pos"])
    X = np.hstack(X).astype(np.float32)
    H = None
    if "ihr" in parts:
        h = np.load(ne.RAW_CACHE / "ihr4" / f"{s}.npy")
        med = np.nanmedian(h)
        iqr = np.nanpercentile(h, 75) - np.nanpercentile(h, 25)
        h = (h - med) / (iqr if iqr > 0 else 1.0)
        pad = IHR_CTX * 4
        h = np.concatenate([np.full(pad, np.nan), h[: nep * 120], np.full(pad + max(0, nep * 120 - len(h)), np.nan)])
        idx = (np.arange(nep) * 120)[:, None] + np.arange(120 + 2 * pad)[None, :]
        H = np.nan_to_num(np.clip(h[idx], -6, 6)).astype(np.float32)
    return X, H


def night_labels(frame: pd.DataFrame, s: str, nep: int, by_subj: dict) -> np.ndarray:
    y = np.full(nep, -1, np.int64)
    idx = by_subj[s]
    ep = frame["epoch_index"].to_numpy()[idx]
    ok = ep < nep
    y[ep[ok]] = frame[TARGET].to_numpy()[idx][ok]
    return y


# ---------------------------------------------------------------- models
class IHREncoder(nn.Module):
    def __init__(self, out: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, out, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, h):  # (B, T, L) -> (B, T, out)
        B, T, L = h.shape
        return self.net(h.reshape(B * T, 1, L)).reshape(B, T, -1)


class SeqModel(nn.Module):
    def __init__(self, n_in: int, kind: str, use_ihr: bool, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.ihr = IHREncoder() if use_ihr else None
        n_in += 16 if use_ihr else 0
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.kind = kind
        if kind == "gru":
            self.core = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
            self.head = nn.Linear(2 * hidden, 5)
        else:
            self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d) for d in (1, 2, 4, 8, 16, 32, 64, 128)])
            self.norms = nn.ModuleList([nn.GroupNorm(1, hidden) for _ in self.convs])
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 5)

    def forward(self, x, h=None):
        if self.ihr is not None:
            x = torch.cat([x, self.ihr(h)], -1)
        z = self.inp(x)
        if self.kind == "gru":
            z, _ = self.core(z)
        else:
            z = z.transpose(1, 2)
            for conv, norm in zip(self.convs, self.norms):
                z = z + self.drop(F.relu(norm(conv(z))))
            z = z.transpose(1, 2)
        return self.head(z)


def batches(ids: list[int], size: int, rng):
    ids = list(rng.permutation(ids))
    for i in range(0, len(ids), size):
        yield ids[i: i + size]


def collate(nights: list[int], X: list, H: list, Y: list):
    T = max(len(X[i]) for i in nights)
    xb = np.zeros((len(nights), T, X[nights[0]].shape[1]), np.float32)
    yb = np.full((len(nights), T), -1, np.int64)
    hb = None if H[nights[0]] is None else np.zeros((len(nights), T, H[nights[0]].shape[1]), np.float32)
    for j, i in enumerate(nights):
        xb[j, : len(X[i])] = X[i]
        yb[j, : len(Y[i])] = Y[i]
        if hb is not None:
            hb[j, : len(H[i])] = H[i]
    return torch.from_numpy(xb), (None if hb is None else torch.from_numpy(hb)), torch.from_numpy(yb)


@torch.no_grad()
def predict(model, nights: list[int], X, H) -> list[np.ndarray]:
    model.eval()
    out = []
    for i in nights:
        xb = torch.from_numpy(X[i][None])
        hb = None if H[i] is None else torch.from_numpy(H[i][None])
        out.append(torch.softmax(model(xb, hb), -1)[0].numpy())
    return out


def train_one(kind: str, use_ihr: bool, tr: list[int], va: list[int], X, H, Y, seed: int, max_epochs: int, patience: int,
              batch: int, hidden: int):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ytr = np.concatenate([Y[i] for i in tr])
    ytr = ytr[ytr >= 0]
    counts = np.bincount(ytr, minlength=5).astype(float)
    weight = torch.tensor(counts.sum() / (5 * counts), dtype=torch.float32)
    model = SeqModel(X[tr[0]].shape[1], kind, use_ihr, hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    best, best_state, wait = -1.0, None, 0
    yva = np.concatenate([Y[i] for i in va])
    for ep in range(max_epochs):
        model.train()
        t0 = time.time()
        for nb in batches(tr, batch, rng):
            xb, hb, yb = collate(nb, X, H, Y)
            xb = xb + 0.05 * torch.randn_like(xb)
            logits = model(xb, hb)
            loss = F.cross_entropy(logits.reshape(-1, 5), yb.reshape(-1), weight=weight, ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pv = np.concatenate(predict(model, va, X, H)).argmax(1)
        f1 = f1_score(yva[yva >= 0], pv[yva >= 0], average="macro")
        if f1 > best:
            best, wait = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        ne.log(f"  seed {seed} epoch {ep}: loss {loss.item():.3f} val F1 {f1:.4f} ({time.time() - t0:.0f}s)")
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return model, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16, help="nights per batch")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--tag", default="", help="suffix for the output name (e.g. a repeat with other seeds)")
    args = ap.parse_args()
    kind, *parts = args.variant.split("_")
    name = args.variant + (f"_{args.tag}" if args.tag else "")
    parts = set(parts)
    assert kind in ("gru", "tcn")
    OUT.mkdir(parents=True, exist_ok=True)
    data = ne.sequence_inputs()
    subjects, frame = data["subjects"], data["frame"]
    splits = [argparse.Namespace(train_idx=a, test_idx=b) for a, b in data["splits"]]
    y = frame[TARGET].to_numpy()
    folds = [np.asarray(sp.test_idx) for sp in splits]
    subj = frame[SUBJ].astype(str).to_numpy()
    by_subj = {s: np.flatnonzero(subj == s) for s in subjects}
    sid = {s: i for i, s in enumerate(subjects)}
    Xraw, H, Y = [], [], []
    for s in subjects:
        x, h = night_inputs(s, parts, data["blocks"])
        Xraw.append(x)
        H.append(h)
        Y.append(night_labels(frame, s, len(x), by_subj))
    ne.log(f"{args.variant}: {len(subjects)} nights, {Xraw[0].shape[1]} inputs per epoch, ihr={'ihr' in parts}")

    proba = np.zeros((len(frame), 5))
    post = np.zeros((len(frame), 5))
    done = np.zeros(len(frame), bool)
    seeds = [int(v) for v in args.seeds.split(",")]
    t_all = time.time()
    for k in [int(v) for v in args.folds.split(",")]:
        sp = splits[k]
        tr_s = sorted(set(subj[np.asarray(sp.train_idx)]))
        te_s = sorted(set(subj[np.asarray(sp.test_idx)]))
        rng = np.random.default_rng(ne.SEED + k)
        va_s = sorted(rng.choice(tr_s, size=max(1, len(tr_s) // 10), replace=False))
        fit_s = [s for s in tr_s if s not in set(va_s)]
        allx = np.vstack([Xraw[sid[s]] for s in fit_s])
        mu, sd = np.nanmean(allx, 0), np.nanstd(allx, 0) + 1e-6
        X = [np.nan_to_num(np.clip((x - mu) / sd, -6, 6)).astype(np.float32) for x in Xraw]
        P = {s: np.zeros((len(X[sid[s]]), 5)) for s in te_s}
        for seed in seeds:
            model, best = train_one(kind, "ihr" in parts, [sid[s] for s in fit_s], [sid[s] for s in va_s], X, H, Y,
                                    seed + 100 * k, args.max_epochs, args.patience, args.batch, args.hidden)
            ne.log(f"fold {k} seed {seed}: best val F1 {best:.4f}")
            for s, p in zip(te_s, predict(model, [sid[s] for s in te_s], X, H)):
                P[s] += p / len(seeds)
        te = np.asarray(sp.test_idx)
        for s in te_s:
            idx = by_subj[s]
            proba[idx] = P[s][frame["epoch_index"].to_numpy()[idx]]
        post[te] = ne.hmm_decode(frame.iloc[np.asarray(sp.train_idx)], frame.iloc[te], proba[te])
        done[te] = True
        ne.log(f"fold {k}: epoch-only {f1_score(y[te], proba[te].argmax(1), average='macro'):.4f}, "
               f"+HMM {f1_score(y[te], post[te].argmax(1), average='macro'):.4f}")
    run_folds = [f for f in folds if done[f].all()]
    out = []
    for decoder, Pm in (("epoch-only", proba), ("hmm", post)):
        row = {"date": time.strftime("%Y-%m-%d"), "variant": name, "decoder": decoder, "seeds": args.seeds,
               "n_folds": len(run_folds), **ne.scores(y[done], Pm[done].argmax(1),
                                                       [np.flatnonzero(np.isin(np.flatnonzero(done), f)) for f in run_folds]),
               "minutes": (time.time() - t_all) / 60}
        out.append(row)
        print(ne.table_line(row["date"], name, decoder, row), flush=True)
    path = OUT / "summary.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(out)], ignore_index=True).to_csv(path, index=False)
    if done.all():
        np.savez_compressed(OUT / f"proba_{name}.npz", proba=proba.astype(np.float32), post=post.astype(np.float32))
    ne.log("done")


if __name__ == "__main__":
    main()
