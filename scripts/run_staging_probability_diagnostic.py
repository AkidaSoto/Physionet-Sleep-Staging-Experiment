
"""Audit feature-model errors, probability targets and night-history cues."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
import warnings

os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import f1_score, roc_auc_score
from threadpoolctl import threadpool_limits
import torch

from run_staging_feature_context import FeatureContextNet, CONTEXTS, prepare_recordings, score_fold
from run_staging_feature_tuning import BASE, TRACKS, FEATURES, profile_masks, save_frame
from run_waveform_development import (
    CACHE, EXPLORATION, KEYS, LABEL, PROBABILITIES, STAGES,
    atomic_json, development_frame, fold_frames, local_training_slot, log, write_predictions,
)

OUT = EXPLORATION/"cinc-probability-diagnostic"
CONTEXT = EXPLORATION/"cinc-feature-context"
RAW = EXPLORATION/"cinc-waveform/scratch/predictions.parquet"
FULL_KEYS = [*KEYS,LABEL,"fold"]
CLASSES = np.arange(5)
SIGNALS = [name.split(".")[-1] for name in TRACKS+FEATURES]
GROUPS = {
    "slow_wave_delta": ["delta_power","frontal_slow_wave_fraction"],
    "alpha_theta": ["c_rel_alpha","o_rel_alpha","occipital_rel_alpha","c_rel_theta","rel_theta"],
    "beta_spindle": ["c_rel_beta","rel_beta","spindle_bursts"],
    "eye_chin": ["eog_high","eye_movement_activity","eye_movement_baseline",
                 "eog_rem_vs_noise_db","eog_rem_vs_sem_db","emg_floor_db"],
    "broad_eeg": ["c_log_power","aperiodic_exponent"],
}
CLOCK = ["elapsed_hours","elapsed_squared","fraction_elapsed"]
PHYSIOLOGY = ["slow_burden_vs_clock","slow_coverage_prior_30min","atonia_fraction_10min","nearby_eye_atonia"]
VARIANTS = {"hard":450, "soft":450, "clock":453, "physiology":457}
SLOW = "topography_features.frontal_slow_wave_fraction"
CHIN = "epoch_ratio_features.emg_floor_db"
EYE = "epoch_ratio_features.eog_rem_vs_noise_db"


def align(frame: pd.DataFrame, path: Path) -> np.ndarray:
    pred = pd.read_parquet(path,columns=[*FULL_KEYS,*PROBABILITIES])
    joined = frame[FULL_KEYS].merge(pred,on=FULL_KEYS,validate="one_to_one")
    if len(joined)!=len(frame) or len(joined)!=len(pred):
        raise ValueError(f"Prediction keys differ at {path}")
    p = joined[PROBABILITIES].to_numpy(np.float64)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError(f"Invalid probabilities at {path}")
    return p


def error_types(frame: pd.DataFrame, tree: np.ndarray, blend: np.ndarray) -> pd.DataFrame:
    y = frame[LABEL].to_numpy()
    tree_class, blend_class = tree.argmax(1), blend.argmax(1)
    repaired = (tree_class!=y)&(blend_class==y)
    regressed = (tree_class==y)&(blend_class!=y)
    history = []
    for _,night in frame.groupby("subject_id",sort=False):
        t,y_n = night.epoch_index.to_numpy(), night[LABEL].to_numpy()
        quarter, interior, boundary = profile_masks(night)
        history.append(pd.DataFrame({"quarter":quarter,"interior":interior,"boundary":boundary>=0},index=night.index))
    result = pd.concat(history).reindex(frame.index)
    result["tree"] = tree_class
    result["blend"] = blend_class
    result["outcome"] = np.select([repaired,regressed,(tree_class!=y)&(blend_class!=y)],
                                  ["repaired","regressed","persistent"],"both_correct")
    result["rival"] = np.where(regressed,blend_class,tree_class)
    result["margin"] = np.sort(tree,axis=1)[:,-1]-np.sort(tree,axis=1)[:,-2]
    for name,pred in [("tree",tree_class),("blend",blend_class)]:
        flags = np.zeros(len(frame),bool)
        for _,night in frame.groupby("subject_id",sort=False):
            at = night.index.to_numpy()
            epoch = night.epoch_index.to_numpy()
            ids = np.cumsum(np.r_[True,(np.diff(epoch)!=1)|(np.diff(y[at])!=0)|(np.diff(pred[at])!=0)])
            flags[at] = (pred[at]!=y[at]) & (np.bincount(ids)[ids]>=6)
        result[f"{name}_sustained"] = flags
    return result


def night_history(subject: str) -> np.ndarray:
    rows = pd.read_parquet(CACHE/"rows6"/f"{subject}.parquet",columns=["slot",SLOW,CHIN,EYE])
    seconds = len(pd.read_parquet(CACHE/"tracks1s"/f"{subject}.parquet",columns=["second"]))
    n = -(-seconds//30)
    observed = np.full((n*5,3),np.nan)
    indexed = rows.loc[rows.slot.between(0,len(observed)-1)]
    observed[indexed.slot.to_numpy()] = indexed[[SLOW,CHIN,EYE]].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        slow,chin,eye = np.nanmean(observed.reshape(n,5,3),axis=1).T
    t = np.arange(n)
    valid_chin = np.isfinite(chin)
    valid_eye = np.isfinite(eye)
    q_chin = np.nanquantile(chin,.25) if valid_chin.any() else -np.inf
    q_eye = np.nanquantile(eye,.75) if valid_eye.any() else np.inf
    atonia = valid_chin & (chin<=q_chin)
    anchor = np.flatnonzero(atonia & valid_eye & (eye>q_eye))
    end_before = np.maximum(0,t-2)
    start_after = np.minimum(n,t+3)
    slow_sum = np.r_[0,np.nancumsum(np.clip(slow,0,1))]
    slow_n = np.r_[0,np.cumsum(np.isfinite(slow))]
    since = np.maximum(0,end_before-60)
    count = slow_n[end_before]-slow_n[since]
    prior_slow = np.divide(slow_sum[end_before]-slow_sum[since],count,out=np.zeros(n),where=count>0)
    burden = slow_sum[end_before]/slow_sum[-1]-end_before/n if slow_sum[-1]>0 else np.zeros(n)
    chin_end,chin_start = np.minimum(n,t+11),np.maximum(0,t-10)
    low_n = np.r_[0,np.cumsum(atonia)]
    total_n = np.r_[0,np.cumsum(valid_chin)]
    left = low_n[end_before]-low_n[chin_start]
    right = low_n[chin_end]-low_n[start_after]
    count_chin = total_n[end_before]-total_n[chin_start]+total_n[chin_end]-total_n[start_after]
    persistence = np.divide(left+right,count_chin,out=np.zeros(n),where=count_chin>0)
    prev = np.r_[-10*n,anchor]
    nex = np.r_[anchor,10*n]
    left_gap = t-prev[np.searchsorted(anchor,end_before,side="left")]
    right_gap = nex[np.searchsorted(anchor,start_after,side="left")]-t
    close_anchor = 1-np.minimum(np.minimum(left_gap,right_gap),60)/60
    hours = t/120
    return np.column_stack([hours,hours**2,t/max(1,n-1),burden,prior_slow,persistence,close_anchor]).astype(np.float32)


def probability_scores(y,p):
    truth = np.eye(5)[y]
    q = p[np.arange(len(y)),y]
    return {"log_loss":float(-np.log(q.clip(1e-12)).mean()),"brier":float(np.square(p-truth).sum(1).mean()),
            "macro_f1":float(f1_score(y,p.argmax(1),labels=CLASSES,average="macro",zero_division=0)),
            "top_two":float((np.argsort(-p,axis=1,kind="stable")[:,:2]==y[:,None]).any(1).mean())}


def temperatures(inner_p: np.ndarray, inner_y: np.ndarray, test_p: np.ndarray):
    logits = np.log(inner_p.clip(1e-10))
    def loss(log_t):
        x = logits/np.exp(log_t)
        return float((logsumexp(x,axis=1)-x[np.arange(len(inner_y)),inner_y]).mean())
    res = minimize_scalar(loss,bounds=(np.log(.25),np.log(4)),method="bounded")
    t = float(np.exp(res.x))
    output = softmax(np.log(test_p.clip(1e-10))/t,axis=1)
    np.testing.assert_array_equal(output.argmax(1),test_p.argmax(1))
    return t,output


def center_inputs(frame,recordings,night_extras):
    matrix = np.empty((len(frame),457),dtype=np.float32)
    for subject,at in frame.groupby("subject_id",sort=False).indices.items():
        x = recordings[subject].features.numpy()
        index = frame.iloc[at].epoch_index.to_numpy()
        window = np.pad(x,((2,2),(0,0)))
        matrix[at,:450] = window[index[:,None]+np.arange(5)].reshape(len(at),450)
        matrix[at,450:] = night_extras[subject][index]
    return matrix


def additive(x,targets,z):
    # One convex multinomial regression with an L2-penalized coefficient matrix.
    n,k = x.shape
    alpha = .001
    def gradient(vector):
        weights = vector.reshape(k+1,5).astype(np.float32)
        q = x@weights[:-1]+weights[-1]
        q -= q.max(1,keepdims=True)
        np.exp(q,out=q)
        q /= q.sum(1,keepdims=True)
        loss = -np.sum(targets*np.log(np.maximum(q,1e-15)),dtype=np.float64)/n
        residual = (q-targets)/n
        slopes = x.T@residual + alpha*weights[:-1]
        grad = np.vstack([slopes,residual.sum(0)])
        return loss + alpha*.5*np.square(weights[:-1]).sum(dtype=np.float64), grad.ravel().astype(np.float64)
    opt = minimize(gradient,np.zeros((k+1)*5),method="L-BFGS-B",jac=True,
                   options={"maxiter":160,"ftol":1e-8,"gtol":3e-5})
    weights = opt.x.reshape(k+1,5).astype(np.float32)
    pred = softmax(z@weights[:-1]+weights[-1],axis=1)
    return pred,weights,{"converged":bool(opt.success),"iterations":int(opt.nit),
                         "max_gradient":float(np.abs(opt.jac).max()),"message":str(opt.message)}


def interventions(teacher,recordings,test,p,indices,out):
    odds = lambda a,b: np.log(a.clip(1e-10))-np.log(b.clip(1e-10))
    case = test.iloc[indices]
    rows = case.index.to_numpy()
    y,rival = case[LABEL].to_numpy(),case.rival.to_numpy()
    orig = odds(p[indices,y],p[indices,rival])
    subjects = sorted(case.subject_id.unique())
    summaries = []
    for group,names in GROUPS.items():
        positions = [i for i,name in enumerate(SIGNALS) if name in names]
        assert len(positions)==len(names)
        original = {s:recordings[s].features for s in subjects}
        for s in subjects:
            changed = original[s].clone().reshape(-1,5,18)
            changed[:,:,positions] = 0.0
            recordings[s].features = changed.reshape(-1,90)
        altered = score_fold(teacher,recordings,case)
        for s in subjects:
            recordings[s].features = original[s]
        changed_odds = odds(altered[np.arange(len(case)),y],altered[np.arange(len(case)),rival])
        summaries.append(pd.DataFrame({"row":rows,"group":group,"log_odds_drop":orig-changed_odds}))
    output = pd.concat(summaries,ignore_index=True)
    save_frame(out,output)


def run_fold(f,frame,split,extras,tree,long,outcomes):
    directory = OUT/f"fold-{f}"
    directory.mkdir(exist_ok=True)
    train,val,test = fold_frames(frame,split)
    test = test.join(outcomes[["outcome","rival"]])
    recordings = prepare_recordings(frame,set(train.subject_id))
    teacher = FeatureContextNet(CONTEXTS["long"])
    state = torch.load(CONTEXT/f"fold-{f}/long/checkpoint.pt",map_location="cpu",weights_only=True)
    teacher.load_state_dict(state["best_weights"])
    test_p = score_fold(teacher,recordings,test)
    if np.max(np.abs(test_p-long[test.index]))>2e-6:
        raise ValueError("Frozen teacher failed exact-key probability reproduction")
    if not (directory/"calibrated.parquet").exists():
        t,calibrated = temperatures(score_fold(teacher,recordings,val),val[LABEL].to_numpy(),test_p)
        write_predictions(directory/"calibrated.parquet",test,calibrated,f)
        counts = np.bincount(train[LABEL],minlength=5)
        corrected_tree = softmax(np.log(tree[test.index].clip(1e-10))+np.log(counts),axis=1)
        write_predictions(directory/"tree_prior.parquet",test,corrected_tree,f)
        atomic_json(directory/"temperature.json",{"t":t})
    if not (directory/"interventions.parquet").exists():
        indices = np.flatnonzero(test.outcome.isin(["repaired","regressed","persistent"]).to_numpy())
        interventions(teacher,recordings,test,test_p,indices,directory/"interventions.parquet")
    if all((directory/f"{v}.parquet").is_file() for v in VARIANTS):
        return
    x,z = center_inputs(train,recordings,extras),center_inputs(test,recordings,extras)
    mu = x.mean(0,dtype=np.float64).astype(np.float32)
    sigma = x.std(0,dtype=np.float64).astype(np.float32)
    sigma[sigma<1e-6] = 1
    x -= mu
    z -= mu
    x /= sigma
    z /= sigma
    np.clip(x,-10,10,out=x)
    np.clip(z,-10,10,out=z)
    labels = train[LABEL].to_numpy()
    counts = np.bincount(labels,minlength=5)
    hard = np.eye(5,dtype=np.float32)[labels]
    target = score_fold(teacher,recordings,train)
    soft = .5*hard+.5*softmax(np.log(target.clip(1e-10))+.70*np.log(counts),axis=1).astype(np.float32)
    del recordings
    for name,k in VARIANTS.items():
        if (directory/f"{name}.parquet").exists():
            continue
        start = perf_counter()
        p,w,diagnostic = additive(x[:,:k],soft if name=="soft" else hard,z[:,:k])
        np.savez_compressed(directory/f"{name}_coefficients.npz",weights=w,means=mu[:k],scales=sigma[:k],
                            columns=np.array([f"{s}@{offset:+d}s" for offset in range(-60,90,6) for s in SIGNALS]+CLOCK+PHYSIOLOGY)[:k])
        atomic_json(directory/f"{name}_optimizer.json",diagnostic)
        write_predictions(directory/f"{name}.parquet",test,p,f)
        log(f"fold {f+1} {name}: {diagnostic['iterations']} iterations, {perf_counter()-start:.1f}s, "
            f"log loss {probability_scores(test[LABEL].to_numpy(),p)['log_loss']:.4f}")


def original_scores(frame,tree,long,raw,outcomes):
    meta = pd.concat([frame[FULL_KEYS],outcomes],axis=1)
    y = frame[LABEL].to_numpy()
    meta["raw_correct"] = raw.argmax(1)==y
    meta["feature_sequence_correct"] = long.argmax(1)==y
    meta["tree_top2_truth"] = (np.argsort(-tree,axis=1,kind="stable")[:,:2]==y[:,None]).any(1)
    totals = meta.groupby("outcome").agg(epochs=("outcome","size"),raw_correct=("raw_correct","sum"),
          sequence_correct=("feature_sequence_correct","sum"),tree_top2_truth=("tree_top2_truth","sum"),
          boundary=("boundary","sum"),interior=("interior","sum"),tree_sustained=("tree_sustained","sum"),
          blend_sustained=("blend_sustained","sum"),median_tree_margin=("margin","median"))
    uncertain = (meta.outcome=="persistent")&(meta.tree!=y)&(~meta.feature_sequence_correct)
    pulse = {"both_feature_models_and_blend_wrong":int(uncertain.sum()),
             "waveform_correct_among_them":int((uncertain&meta.raw_correct).sum())}
    directions = meta.loc[meta.outcome.ne("both_correct")].groupby(["outcome",LABEL,"rival"],as_index=False).agg(
        epochs=("outcome","size"),raw_correct=("raw_correct","sum"),boundary=("boundary","sum"),
        interior=("interior","sum"),tree_sustained=("tree_sustained","sum"),blend_sustained=("blend_sustained","sum"))
    directions.to_csv(OUT/"error_directions.csv",index=False)
    save_frame(OUT/"error_cases.parquet",meta)
    return totals.reset_index().to_dict("records"),pulse


def summarize(frame,predictions,outcomes):
    y = frame[LABEL].to_numpy()
    metrics = {}
    for name,p in predictions.items():
        metrics[name] = {"pooled":probability_scores(y,p),
                         "folds":[probability_scores(y[frame.fold.eq(f)],p[frame.fold.eq(f)]) for f in range(5)]}
    pairs = [(2,3),(4,1),(4,2),(1,0)]
    pair_rows = []
    for a,b in pairs:
        base = (y==a)|(y==b)
        for cohort,mask in [("all",base),("interior",base&outcomes.interior.to_numpy())]:
            for name in VARIANTS:
                p = predictions[name]
                f_values = []
                for f in range(5):
                    rows = mask&frame.fold.eq(f).to_numpy()
                    z = p[rows,a]/(p[rows,a]+p[rows,b]).clip(1e-10)
                    f_values.append(float(roc_auc_score(y[rows]==a,z)))
                h = p[mask,a]/(p[mask,a]+p[mask,b]).clip(1e-10)
                truth = y[mask]==a
                pair_rows.append({"stage_a":STAGES[a],"stage_b":STAGES[b],"cohort":cohort,"variant":name,
                                  "examples":int(mask.sum()),"mean_fold_auc":float(np.mean(f_values)),
                                  "fold_auc":f_values,"binary_brier":float(np.square(truth-h).mean())})
    pd.DataFrame(pair_rows).to_csv(OUT/"pair_probability_quality.csv",index=False)
    return metrics,pair_rows


def execute():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    frame,splits = development_frame()
    frame["fold"] = -1
    for i,split in enumerate(splits):
        frame.iloc[split.test_idx,frame.columns.get_loc("fold")] = i
    # development_frame order is retained for split positions.
    tree,long,raw = [align(frame,p) for p in (BASE,CONTEXT/"long_predictions.parquet",RAW)]
    blend = .5*tree+.5*long
    order = frame.sort_values(KEYS).index
    ordered = frame.loc[order]
    # Compute run structure in recording order and restore the original frame positions.
    ordering = ordered.reset_index(drop=True)
    outcomes = error_types(ordering,tree[order],blend[order]).set_axis(order).reindex(frame.index)
    totals,residual = original_scores(frame,tree,long,raw,outcomes)
    fixed = {"tree":tree,"sequence":long,"blend":blend,"waveform":raw}
    atomic_json(OUT/"archived.json",{"totals":totals,"residual":residual,"probability":{k:probability_scores(frame[LABEL].to_numpy(),p)
                                                                                  for k,p in fixed.items()}})
    extras = {s:night_history(s) for s in sorted(frame.subject_id.unique())}
    for f,split in enumerate(splits):
        log(f"fold {f+1}: probability audit and additive controls")
        run_fold(f,frame,split,extras,tree,long,outcomes)
    predictions = dict(fixed)
    for v in [*VARIANTS,"calibrated","tree_prior"]:
        epochs = pd.concat([pd.read_parquet(OUT/f"fold-{f}/{v}.parquet") for f in range(5)])
        p = frame[FULL_KEYS].merge(epochs,on=FULL_KEYS,validate="one_to_one")
        if len(p)!=len(frame) or len(epochs)!=len(frame):
            raise ValueError("Incomplete student/calibrated predictions")
        predictions[v] = p[PROBABILITIES].to_numpy()
        save_frame(OUT/f"{v}_predictions.parquet",p)
    metrics,pairs = summarize(frame,predictions,outcomes)
    changes = pd.concat([pd.read_parquet(OUT/f"fold-{f}/interventions.parquet") for f in range(5)])
    attrs = changes.join(frame[[LABEL,"fold"]],on="row").join(outcomes[["outcome","rival"]],on="row")
    coalition = attrs.groupby(["outcome",LABEL,"rival","group"],as_index=False).log_odds_drop.agg(
        epochs="size",median="median",mean="mean")
    coalition.to_csv(OUT/"feature_group_sensitivity.csv",index=False)
    atomic_json(OUT/"summary.json",{"probability":metrics,"pairs":pairs,"totals":totals,"residual":residual})


if __name__=="__main__":
    if not (OUT/"FINDINGS.md").exists():
        raise FileNotFoundError("Declare experiments before running")
    with local_training_slot(), threadpool_limits(limits=4):
        execute()

