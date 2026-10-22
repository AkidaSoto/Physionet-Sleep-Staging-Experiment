"""Pre-declared CV experiment on the adopted window model (nocardio27): prune six inputs, test spindle-burst measures.

Variants (see artifacts/staging_window_model/cinc2018/feature_reduction/FINDINGS.md):
  prune6             nocardio27 minus 3 residual-power features and tracks emg_log_rms, eog_low, c_rel_delta
  prune6_sigratio    prune6 with c_rel_sigma replaced by c_sigratio = sigma / (alpha + beta), 1-s track
  prune6_bursts      prune6 + spindle_bursts: seconds per epoch with the ratio > night median + 2 robust SDs
  nocardio27_bursts  nocardio27 + spindle_bursts
Development subjects only, same folds / seed / scaling / encoder as run_window_hmm_staging.py; epoch-only (no decoder).
Run one variant per process (memory): python scripts/run_spindle_feature_experiment.py --variant prune6
Effect sizes: python scripts/run_spindle_feature_experiment.py --effect-sizes
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_window_hmm_staging as w  # noqa: E402

from physionet_sleep.analysis.epoch_windows import (  # noqa: E402
    EPOCH_SECONDS, SUBJ, TARGET, WINDOW_FEATURES, WindowEncoder, dense_signals, epoch_frame, robust_scale,
)
from physionet_sleep.experiments.tabular_runner import build_subject_eval_splits  # noqa: E402

OUT = w.REPO / "artifacts/staging_window_model/cinc2018/feature_reduction"
PRUNE_FEATURES = ["spindle_features.theta_residual_power", "spindle_features.beta_residual_power",
                  "spindle_features.alpha_residual_power"]
PRUNE_TRACKS = ["emg_log_rms", "eog_low", "c_rel_delta"]
SIGRATIO, BURSTS = "c_sigratio", "spindle_bursts"
KC, EOG_RESID = "kc_transients", "eog_fast_resid"
SAWTOOTH, ALPHA = "sawtooth_theta", "alpha_dropout"


def variant(name: str) -> tuple[list[str], list[str], list[str]]:
    """(tracks, features, extra signals) for a variant; extras are appended after tracks + features."""
    trk, feats = w.nocardio27_signals()
    if name == "nocardio27":
        return trk, feats, []
    if name == "nocardio27_bursts":
        return trk, feats, [BURSTS]
    trk = [t for t in trk if t not in PRUNE_TRACKS]
    feats = [f for f in feats if f not in PRUNE_FEATURES]
    if name == "prune6":
        return trk, feats, []
    if name == "prune6_sigratio":
        return [t for t in trk if t != "c_rel_sigma"], feats, [SIGRATIO]
    if name == "prune6_bursts":
        return trk, feats, [BURSTS]
    # round 4 step B: sawtooth_theta (the only candidate that passed validation)
    if name == "swap4_saw":
        return trk, [f for f in feats if f != "spindle_features.gamma_residual_power"], [SAWTOOTH]
    if name == "swap4_alpha":
        return [t for t in trk if t != "o_rel_alpha"], feats, [ALPHA]
    if name == "swap4_combo":  # both passing round-4 swaps (the only passing round-3 swap drops aperiodic_intercept)
        return ([t for t in trk if t != "o_rel_alpha"], [f for f in feats if f != "spindle_features.gamma_residual_power"],
                [SAWTOOTH, ALPHA])
    # round 3: validated spindle measures only
    s1 = ["spindle_features.gamma_residual_power", "eye_movement_activity.eye_movement_baseline"]
    if name in ("swapS1", "swapS3"):
        feats = [f for f in feats if f not in s1]
        return ([t for t in trk if t != "c_rel_sigma"], feats, [SIGRATIO, BURSTS]) if name == "swapS3" else (trk, feats, [BURSTS])
    if name == "swapS2":
        s2 = ["spindle_features.aperiodic_intercept", "spindle_features.gamma_residual_power"]
        return trk, [f for f in feats if f not in s2], [BURSTS]
    # round 2: swaps on prune6
    drop_a, drop_b = ["spindle_features.aperiodic_intercept", "spindle_features.gamma_residual_power"], \
        ["eye_movement_activity.eye_movement_baseline"]
    a, b = name in ("swapA", "swapAB", "dropAB"), name in ("swapB", "swapAB", "dropAB")
    if not (a or b):
        raise ValueError(name)
    feats = [f for f in feats if not (a and f in drop_a) and not (b and f in drop_b)]
    trk = [t for t in trk if not (b and t == "eog_high")]
    extra = [] if name == "dropAB" else ([BURSTS, KC] if a else []) + ([EOG_RESID] if b else [])
    return trk, feats, extra


def sigma_ratio(tracks: pd.DataFrame) -> np.ndarray:
    return (tracks["c_rel_sigma"] / (tracks["c_rel_alpha"] + tracks["c_rel_beta"] + 1e-6)).to_numpy(np.float32)


def burst_seconds(ratio: np.ndarray) -> np.ndarray:
    """1-s indicator: ratio above the night's median + 2 x 1.4826 x MAD."""
    med = np.nanmedian(ratio)
    mad = 1.4826 * np.nanmedian(np.abs(ratio - med))
    return (ratio > med + 2 * mad).astype(np.float32)


def epoch_bursts(ratio: np.ndarray) -> np.ndarray:
    """Burst count per 30-s epoch (epoch k = seconds 30k..30k+29), one value per epoch."""
    b = burst_seconds(ratio)
    n = -(-len(b) // EPOCH_SECONDS)
    b = np.r_[b, np.zeros(n * EPOCH_SECONDS - len(b), np.float32)]
    return b.reshape(n, EPOCH_SECONDS).sum(1)


def per_epoch_count(b: np.ndarray) -> np.ndarray:
    n = -(-len(b) // EPOCH_SECONDS)
    return np.r_[b, np.zeros(n * EPOCH_SECONDS - len(b), np.float32)].reshape(n, EPOCH_SECONDS).sum(1)


def kc_seconds(delta: np.ndarray, half: int = 15) -> np.ndarray:
    """1-s indicator: c_rel_delta above its centred +-15 s rolling median by > 2.5 rolling robust SDs."""
    x = np.r_[np.full(half, np.nan), delta, np.full(half, np.nan)].astype(np.float32)
    v = np.lib.stride_tricks.sliding_window_view(x, 2 * half + 1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(v, 1)
        sd = 1.4826 * np.nanmedian(np.abs(v - med[:, None]), 1)
        return ((delta - med) > 2.5 * sd).astype(np.float32) * (sd > 0)


def epoch_kc(tracks: pd.DataFrame) -> np.ndarray:
    return per_epoch_count(kc_seconds(tracks["c_rel_delta"].to_numpy(np.float32)))


def eog_residual(tracks: pd.DataFrame) -> np.ndarray:
    """eog_high minus its per-night least-squares fit on f_rel_delta and c_log_power."""
    y = tracks["eog_high"].to_numpy(float)
    A = np.c_[np.ones(len(y)), tracks["f_rel_delta"].to_numpy(float), tracks["c_log_power"].to_numpy(float)]
    ok = np.isfinite(y) & np.isfinite(A).all(1)
    coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
    return (y - A @ coef).astype(np.float32)


def extra_column(name: str, tracks: pd.DataFrame) -> np.ndarray:
    """(n_seconds,) subject-scaled extra signal."""
    ratio = sigma_ratio(tracks)
    if name == SIGRATIO:
        x = ratio
    elif name == EOG_RESID:
        x = eog_residual(tracks)
    elif name in (SAWTOOTH, ALPHA):
        x = np.repeat(round4_measures(tracks)[name], EPOCH_SECONDS)[: len(ratio)]
    else:  # epoch value held across the epoch's seconds
        ep = epoch_bursts(ratio) if name == BURSTS else epoch_kc(tracks)
        x = np.repeat(ep, EPOCH_SECONDS)[: len(ratio)]
    return robust_scale(x[:, None], [name])[:, 0]


def cohens_d(x: np.ndarray, is_n2: np.ndarray) -> float:
    a, b = x[is_n2 & np.isfinite(x)], x[~is_n2 & np.isfinite(x)]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp)


def effect_sizes(subjects, rows, tracks, frame) -> None:
    """N2 one-vs-rest Cohen's d of the epoch-level spindle measures (raw and per-subject scaled)."""
    per = []
    for s in subjects:
        t = tracks[s]
        ratio = sigma_ratio(t)
        n = -(-len(ratio) // EPOCH_SECONDS)
        def ep_mean(x):
            x = np.r_[x, np.full(n * EPOCH_SECONDS - len(x), np.nan, np.float32)]
            with np.errstate(all="ignore"):
                return np.nanmean(x.reshape(n, EPOCH_SECONDS), 1)
        d = pd.DataFrame({"c_rel_sigma": ep_mean(t["c_rel_sigma"].to_numpy(np.float32)),
                          SIGRATIO: ep_mean(ratio), BURSTS: epoch_bursts(ratio)})
        d["epoch_index"], d[SUBJ] = np.arange(n), s
        per.append(d)
    ep = pd.concat(per).merge(frame[[SUBJ, "epoch_index", TARGET]], on=[SUBJ, "epoch_index"])
    is_n2 = ep[TARGET].to_numpy() == 2
    out = []
    for c in ["c_rel_sigma", SIGRATIO, BURSTS]:
        g = ep.groupby(SUBJ)[c]
        iqr = (g.transform(lambda s: s.quantile(0.75) - s.quantile(0.25))).replace(0, np.nan).fillna(1.0)
        scaled = ((ep[c] - g.transform("median")) / iqr).to_numpy(float)
        row = {"measure": c, "d_N2_raw": cohens_d(ep[c].to_numpy(float), is_n2), "d_N2_subject_scaled": cohens_d(scaled, is_n2)}
        for k, st in enumerate(w.STAGES):
            row[f"mean_{st}"] = float(ep.loc[ep[TARGET] == k, c].mean())
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(OUT / "effect_sizes.csv", index=False)
    print(res.to_string())


def effect_sizes_round2(subjects, tracks, frame) -> None:
    """Cohen's d of the round-2 measures for their target stage against specific stages and the rest."""
    per = []
    for s in subjects:
        t = tracks[s]
        n = -(-len(t) // EPOCH_SECONDS)
        def ep_mean(x):
            x = np.r_[x, np.full(n * EPOCH_SECONDS - len(x), np.nan, np.float32)]
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.nanmean(x.reshape(n, EPOCH_SECONDS), 1)
        per.append(pd.DataFrame({SUBJ: s, "epoch_index": np.arange(n), BURSTS: epoch_bursts(sigma_ratio(t)),
                                 KC: epoch_kc(t), EOG_RESID: ep_mean(eog_residual(t)),
                                 "eog_high": ep_mean(t["eog_high"].to_numpy(np.float32))}))
    ep = pd.concat(per).merge(frame[[SUBJ, "epoch_index", TARGET]], on=[SUBJ, "epoch_index"])
    y = ep[TARGET].to_numpy()
    targets = {BURSTS: (2, {"Wake": 0, "N1": 1, "REM": 4}), KC: (2, {"Wake": 0, "N1": 1, "REM": 4}),
               EOG_RESID: (4, {"N3": 3}), "eog_high": (4, {"N3": 3})}
    out = []
    for c, (tgt, others) in targets.items():
        g = ep.groupby(SUBJ)[c]
        iqr = (g.transform(lambda s: s.quantile(0.75) - s.quantile(0.25))).replace(0, np.nan).fillna(1.0)
        for scale, x in (("raw", ep[c].to_numpy(float)), ("subject_scaled", ((ep[c] - g.transform("median")) / iqr).to_numpy(float))):
            row = {"measure": c, "target": w.STAGES[tgt], "scale": scale}
            for nm, k in others.items():
                m = (y == tgt) | (y == k)
                row[f"d_vs_{nm}"] = cohens_d(x[m], y[m] == tgt)
            row["d_vs_rest"] = cohens_d(x, y == tgt)
            out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(OUT / "effect_sizes_round2.csv", index=False)
    print(res.to_string())


def robust_high(x: np.ndarray, k: float) -> np.ndarray:
    med = np.nanmedian(x)
    return x > med + k * 1.4826 * np.nanmedian(np.abs(x - med))


def rolling_jump(x: np.ndarray, k: float, half: int = 15) -> np.ndarray:
    """x above its centred +-half s rolling median by more than k rolling robust SDs."""
    xp = np.r_[np.full(half, np.nan), x, np.full(half, np.nan)].astype(np.float32)
    v = np.lib.stride_tricks.sliding_window_view(xp, 2 * half + 1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(v, 1)
        sd = 1.4826 * np.nanmedian(np.abs(v - med[:, None]), 1)
    return ((x - med) > k * sd) & (sd > 0)


def residual(y: np.ndarray, *regressors: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(y))] + [np.asarray(r, float) for r in regressors])
    ok = np.isfinite(y) & np.isfinite(A).all(1)
    coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
    return (y - A @ coef).astype(np.float32)


def epoch_mean(x: np.ndarray) -> np.ndarray:
    n = -(-len(x) // EPOCH_SECONDS)
    x = np.r_[np.asarray(x, np.float32), np.full(n * EPOCH_SECONDS - len(x), np.nan, np.float32)]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(x.reshape(n, EPOCH_SECONDS), 1)


def round4_measures(t: pd.DataFrame) -> dict[str, np.ndarray]:
    """Epoch-level round-4 candidates for one night (see FINDINGS.md, round 4 Step A)."""
    g = lambda c: t[c].to_numpy(np.float32)  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        below = lambda c: g(c) < np.nanmedian(g(c))  # noqa: E731
        above = lambda c: g(c) > np.nanmedian(g(c))  # noqa: E731
        chin = epoch_mean(g("emg_log_rms"))
        alpha = epoch_mean(g("o_rel_alpha"))
        ref = np.nanmedian(alpha[chin > np.nanpercentile(chin, 75)])
    return {
        "rem_eye_bursts": per_epoch_count((robust_high(g("eog_high"), 2) & below("f_rel_delta")).astype(np.float32)),
        "kcomplex_events": per_epoch_count((rolling_jump(g("c_log_power"), 2.5) & above("c_rel_delta")).astype(np.float32)),
        "slow_eye_movements": epoch_mean(residual(g("eog_low").astype(float), g("f_rel_delta"), g("c_log_power"))),
        "alpha_dropout": alpha / ref,
        "sawtooth_theta": per_epoch_count((robust_high(g("c_rel_theta"), 2) & below("emg_log_rms")).astype(np.float32)),
    }


ROUND4_TARGETS = {  # measure -> (target stage, comparisons, direction)
    "rem_eye_bursts": (4, ["N3", "N2", "rest"], 1), "kcomplex_events": (2, ["Wake", "N1", "REM"], 1),
    "slow_eye_movements": (1, ["Wake", "N2", "rest"], 1), "alpha_dropout": (1, ["Wake"], -1),
    "sawtooth_theta": (4, ["rest"], 1),
}


def validate_round4() -> None:
    """Step A: Cohen's d of each candidate for its target stage, streaming one night at a time (low memory)."""
    from physionet_sleep.analysis.epoch_windows import subject_epochs
    cache = w.REPO / "artifacts/cinc2018_window_cache"
    per = []
    for s in w.development_subjects("cinc2018"):
        lab = subject_epochs(pd.read_parquet(cache / "rows6" / f"{s}.parquet"), [])[["epoch_index", TARGET]]
        m = pd.DataFrame(round4_measures(pd.read_parquet(cache / "tracks1s" / f"{s}.parquet")))
        m["epoch_index"], m[SUBJ] = np.arange(len(m)), s
        per.append(m.merge(lab, on="epoch_index"))
    ep = pd.concat(per, ignore_index=True)
    y = ep[TARGET].to_numpy()
    out = []
    for c, (tgt, comps, sign) in ROUND4_TARGETS.items():
        g = ep.groupby(SUBJ)[c]
        iqr = (g.transform(lambda s: s.quantile(0.75) - s.quantile(0.25))).replace(0, np.nan).fillna(1.0)
        scaled = ((ep[c] - g.transform("median")) / iqr).to_numpy(float)
        row = {"measure": c, "target": w.STAGES[tgt], "direction": "+" if sign > 0 else "-"}
        for scale, x in (("raw", ep[c].to_numpy(float)), ("scaled", scaled)):
            for cmp_ in comps:
                m = np.ones(len(y), bool) if cmp_ == "rest" else (y == tgt) | (y == w.STAGES.index(cmp_))
                row[f"d_{scale}_vs_{cmp_}"] = cohens_d(x[m], y[m] == tgt)
        row["min_signed_scaled_d"] = min(sign * row[f"d_scaled_vs_{c_}"] for c_ in comps)
        row["passes"] = row["min_signed_scaled_d"] >= 0.5
        for k, st in enumerate(w.STAGES):
            row[f"mean_{st}"] = float(ep.loc[y == k, c].mean())
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(OUT / "round4_validation.csv", index=False)
    print(res.T.to_string())


REPLACED = {  # candidate -> (input it would replace, kind, predicted sign of the replaced input)
    "rem_eye_bursts": ("eye_movement_activity.eye_movement_baseline", "feature", 1),
    "kcomplex_events": ("spindle_features.gamma_residual_power", "feature", 1),
    "slow_eye_movements": ("epoch_ratio_features.eog_rem_vs_sem_db", "feature", -1),
    "alpha_dropout": ("o_rel_alpha", "track", -1),
}


def validate_round4_amended() -> None:
    """Amended gate: candidate d > 0 (signed) for every comparison and >= the replaced input's signed d."""
    from physionet_sleep.analysis.epoch_windows import subject_epochs
    cache = w.REPO / "artifacts/cinc2018_window_cache"
    feats = [v[0] for v in REPLACED.values() if v[1] == "feature"]
    per = []
    for s in w.development_subjects("cinc2018"):
        r = pd.read_parquet(cache / "rows6" / f"{s}.parquet")
        lab = subject_epochs(r, feats)[["epoch_index", TARGET] + feats]
        t = pd.read_parquet(cache / "tracks1s" / f"{s}.parquet")
        m = pd.DataFrame(round4_measures(t))
        m["o_rel_alpha"] = epoch_mean(t["o_rel_alpha"].to_numpy(np.float32))
        m["epoch_index"], m[SUBJ] = np.arange(len(m)), s
        per.append(m.merge(lab, on="epoch_index"))
    ep = pd.concat(per, ignore_index=True)
    y = ep[TARGET].to_numpy()

    def scaled(c):
        g = ep.groupby(SUBJ)[c]
        iqr = (g.transform(lambda s: s.quantile(0.75) - s.quantile(0.25))).replace(0, np.nan).fillna(1.0)
        return ((ep[c] - g.transform("median")) / iqr).to_numpy(float)

    def d(x, tgt, cmp_):
        m = np.ones(len(y), bool) if cmp_ == "rest" else (y == tgt) | (y == w.STAGES.index(cmp_))
        return cohens_d(x[m], y[m] == tgt)

    out = []
    for c, (rep, _, rsign) in REPLACED.items():
        tgt, comps, sign = ROUND4_TARGETS[c]
        xc, xr = scaled(c), scaled(rep)
        for cmp_ in comps:
            dc, dr = sign * d(xc, tgt, cmp_), rsign * d(xr, tgt, cmp_)
            out.append({"candidate": c, "replaced": rep.split(".")[-1], "target": w.STAGES[tgt], "vs": cmp_,
                        "signed_d_candidate": dc, "signed_d_replaced": dr, "ok": (dc > 0) and (dc >= dr)})
    res = pd.DataFrame(out)
    res["passes"] = res.groupby("candidate")["ok"].transform("all")
    res.to_csv(OUT / "round4_validation_amended.csv", index=False)
    print(res.round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant")
    ap.add_argument("--effect-sizes", action="store_true")
    ap.add_argument("--effect-sizes-round2", action="store_true")
    ap.add_argument("--importance", action="store_true", help="per-signal permutation importance on each test fold")
    ap.add_argument("--validate-round4", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.validate_round4:
        validate_round4()
        validate_round4_amended()
        return
    cache = w.REPO / "artifacts/cinc2018_window_cache"
    subjects, rows, tracks = w.load_inputs("cinc2018", cache)  # dev_subjects.txt only
    assert len(subjects) == 320
    frame = epoch_frame(rows, WINDOW_FEATURES)
    w.log(f"{len(subjects)} subjects, {len(frame)} labelled epochs")
    if args.effect_sizes:
        effect_sizes(subjects, rows, tracks, frame)
        return
    if args.effect_sizes_round2:
        effect_sizes_round2(subjects, tracks, frame)
        return

    cfg = args.variant
    trk, feats, extra = variant(cfg)
    names = [c.split(".")[-1] for c in trk + feats] + extra
    w.log(f"{cfg}: {len(trk)} tracks, {len(feats)} features, extra {extra} -> {len(names)} signals")
    dense = {}
    for s in subjects:
        X = dense_signals(tracks[s], rows[s], trk, feats)
        if extra:
            X = np.hstack([X] + [extra_column(e, tracks[s])[: len(X), None] for e in extra])
        dense[s] = X
    del tracks
    splits = list(build_subject_eval_splits(groups=frame[SUBJ].astype(str).to_numpy(), y=frame[TARGET].to_numpy(),
                                            outer_validation="kfold", outer_folds=5, random_state=w.SEED))
    folds = [np.asarray(sp.test_idx) for sp in splits]
    y = frame[TARGET].to_numpy()
    proba = np.zeros((len(frame), 5))
    rng, importance = np.random.default_rng(w.SEED), []
    for k, sp in enumerate(splits):
        tr, te = np.asarray(sp.train_idx), np.asarray(sp.test_idx)
        train, test = frame.iloc[tr], frame.iloc[te]
        enc = WindowEncoder(names, context=2).fit(dense, sorted(set(train[SUBJ])))
        Wtr, Wte = enc.transform(dense, train), enc.transform(dense, test)
        model, p = w.fit_proba(Wtr.reshape(len(Wtr), -1), y[tr], Wte.reshape(len(Wte), -1), enc.column_names())
        del Wtr
        if args.importance:  # shuffle one signal (all steps) across held-out epochs
            cols, base = enc.column_names(), f1_score(y[te], p.argmax(1), average="macro")
            for j, nm in enumerate(names):
                Xp = Wte.copy()
                Xp[:, :, j] = Xp[rng.permutation(len(Xp)), :, j]
                pred_p = model.predict(pd.DataFrame(Xp.reshape(len(Xp), -1), columns=cols))
                importance.append({"config": cfg, "fold": k, "signal": nm, "drop": base - f1_score(y[te], pred_p, average="macro"),
                                   "drop_N2": f1_score(y[te], p.argmax(1), labels=[2], average=None)[0]
                                   - f1_score(y[te], pred_p, labels=[2], average=None)[0]})
                del Xp
        del Wte
        proba[te] = p
        w.log(f"{cfg} fold {k}: {f1_score(y[te], p.argmax(1), average='macro'):.4f}")
    pred = proba.argmax(1)
    row = {"config": cfg, "decoder": "epoch-only", "n_signals": len(names), **w.scores(frame, pred, folds)}
    w.log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    fold_rows = [{"config": cfg, "decoder": "epoch-only", "fold": k, "macro_f1": f1_score(y[i], pred[i], average="macro"),
                  "kappa": w.cohen_kappa_score(y[i], pred[i])} for k, i in enumerate(folds)]
    for fname, new in (("summary.csv", [row]), ("folds.csv", fold_rows)):
        path = OUT / fname
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if len(old):
            old = old[old["config"] != cfg]
        pd.concat([old, pd.DataFrame(new)], ignore_index=True).to_csv(path, index=False)
    np.save(OUT / f"proba_{cfg}.npy", proba.astype(np.float32))
    if importance:
        path = OUT / "importance.csv"
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if len(old):
            old = old[old["config"] != cfg]
        pd.concat([old, pd.DataFrame(importance)], ignore_index=True).to_csv(path, index=False)
    w.log(f"done -> {OUT}")


if __name__ == "__main__":
    main()
