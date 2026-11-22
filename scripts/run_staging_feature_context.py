
"""Run the short/long compact feature-context experiments #47–48."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from run_staging_feature_tuning import BASE, TRACKS, FEATURES, error_structure, save_frame
from run_waveform_development import (
    CACHE, EXPLORATION, KEYS, LABEL, PROBABILITIES, SEED, STAGES,
    atomic_json, development_frame, fold_frames, local_training_slot, log, pin_manifest,
    scores, write_predictions,
)
from build_metric_intervals import cv_fold_draws, cv_scores, interval, patient_confusions
from physionet_sleep.analysis.epoch_windows import WindowEncoder
from physionet_sleep.experiments.feature_context import FeatureContextNet
from physionet_sleep.experiments.modern_tabular import _dense_cached_night
from physionet_sleep.experiments.waveform_clock import active_clock


OUT = EXPLORATION / "cinc-feature-context"
CONTEXTS = {"short": (1, 1, 1, 1), "long": (1, 2, 4, 8)}
BATCH_NIGHTS = 8
MAX_PASSES = 30
CLASSES = np.arange(len(STAGES))


@dataclass
class Recording:
    features: torch.Tensor
    scored_labels: torch.Tensor


def prepare_recordings(frame: pd.DataFrame, train_subjects: set[str]) -> dict[str, Recording]:
    dense = {}
    for subject in sorted(frame.subject_id.unique()):
        dense[subject] = _dense_cached_night(CACHE, subject, TRACKS, FEATURES)
    if len(TRACKS) + len(FEATURES) != 18:
        raise ValueError("Expected the original 18 compact inputs")
    # Streaming fold moments give the same normalization as WindowEncoder
    # without materializing a second full-cohort dense seconds matrix.
    count = np.zeros(18, dtype=np.int64)
    total = np.zeros(18, dtype=np.float64)
    squared = np.zeros(18, dtype=np.float64)
    for s in sorted(train_subjects):
        x = dense[s]
        finite = np.isfinite(x)
        count += finite.sum(axis=0)
        total += np.nansum(x, axis=0, dtype=np.float64)
        squared += np.nansum(np.square(x, dtype=np.float64), axis=0)
    if not np.all(count):
        raise ValueError("Every signal must have observed values in the training patients")
    mean = total / count
    sd = np.sqrt(np.maximum(0, squared/count-mean**2)) + 1e-6
    encoder = WindowEncoder(TRACKS+FEATURES, context=0, mean=mean, sd=sd)
    inputs = {}
    for subject, night in frame.groupby("subject_id", sort=False):
        x = dense.pop(subject)
        pad = encoder.steps_per_epoch
        steps = encoder._steps(x)[pad:-pad]
        steps = np.pad(steps, ((0, -len(steps)%pad),(0,0)))
        values = steps.reshape(-1, 90)
        y = np.full(len(values), -1, dtype=np.int64)
        at = night.epoch_index.to_numpy()
        if at.max() >= len(values):
            raise ValueError(f"Scored epoch beyond the recorded input for {subject}")
        y[at] = night[LABEL]
        inputs[subject] = Recording(torch.from_numpy(values), torch.from_numpy(y))
    return inputs


def batches(subjects: list[str], rng: np.random.Generator | None = None):
    order = subjects if rng is None else rng.permutation(subjects)
    for first in range(0, len(order), BATCH_NIGHTS):
        yield order[first:first + BATCH_NIGHTS]


def collate(recordings: dict[str, Recording], subjects) -> tuple[torch.Tensor, ...]:
    x = [recordings[s].features for s in subjects]
    y = [recordings[s].scored_labels for s in subjects]
    x_pad = pad_sequence(x, batch_first=True)
    y_pad = pad_sequence(y, batch_first=True, padding_value=-1)
    mask = torch.arange(x_pad.shape[1])[None, :] < torch.tensor([len(row) for row in x])[:, None]
    return x_pad, y_pad, mask


@torch.inference_mode()
def validation_score(model, recordings: dict[str, Recording], subjects: list[str]) -> float:
    model.eval()
    truth, predicted = [], []
    for batch in batches(subjects):
        x, y, mask = collate(recordings, batch)
        scored = y >= 0
        truth.append(y[scored].numpy())
        predicted.append(model(x, mask).argmax(-1)[scored].numpy())
    return float(f1_score(np.concatenate(truth), np.concatenate(predicted), labels=CLASSES,
                          average="macro", zero_division=0))


@torch.inference_mode()
def score_fold(model, recordings: dict[str, Recording], test: pd.DataFrame) -> np.ndarray:
    model.eval()
    rows = test.groupby("subject_id", sort=False).indices
    result = np.empty((len(test),len(STAGES)), dtype=np.float32)
    for batch in batches(list(rows)):
        x, _, mask = collate(recordings, batch)
        probabilities = model(x, mask).softmax(dim=-1).numpy()
        for b, subject in enumerate(batch):
            positions = rows[subject]
            result[positions] = probabilities[b, test.iloc[positions].epoch_index.to_numpy()]
    return result


def checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".part")
    torch.save(state, tmp)
    tmp.replace(path)


def fit(
    variant: str, recordings: dict[str, Recording], train: list[str], validation: list[str],
    seed: int, directory: Path, pause_path: Path, deadline: float,
    *, smoke_pause_after: int | None = None, max_passes: int = MAX_PASSES,
) -> tuple[FeatureContextNet | None, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "checkpoint.pt"
    torch.manual_seed(seed)
    model = FeatureContextNet(CONTEXTS[variant])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max_passes,eta_min=0.0003)
    y = torch.cat([recordings[s].scored_labels for s in train])
    counts = torch.bincount(y[y>=0],minlength=len(STAGES)).float()
    if torch.any(counts == 0):
        raise ValueError("Missing stage in the patient training partition")
    weight = counts.pow(-0.70)
    state = {"next_pass":0,"best":-1.0,"best_pass":0,"best_weights":None,"history":[]}
    if latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng"])
    start = active_clock()
    is_done = state["next_pass"] >= 12 and state["next_pass"] - state["best_pass"] >= 5
    if not is_done:
        for epoch in range(state["next_pass"], max_passes):
            if pause_path.exists() or active_clock() >= deadline:
                return None,state
            began = active_clock()
            model.train()
            loss_sum = 0.0
            for batch in batches(train,np.random.default_rng(seed+epoch)):
                x, y, mask = collate(recordings,batch)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x,mask)
                loss = F.cross_entropy(logits.flatten(0,1), y.flatten(), weight=weight, ignore_index=-1)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                loss_sum += loss.item()
            scheduler.step()
            metric = validation_score(model,recordings,validation)
            if metric > state["best"]:
                state.update(best=metric,best_pass=epoch+1,
                             best_weights={key:value.detach().clone() for key,value in model.state_dict().items()})
            state["history"].append({"pass":epoch+1,"val_macro_f1":metric,
                                     "batch_loss_sum":loss_sum,"seconds":active_clock()-began})
            state.update(next_pass=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(),rng=torch.get_rng_state())
            checkpoint(latest,state)
            if (epoch+1)%5 == 0 or epoch==0:
                log(f"{directory.parent.name} {variant} pass {epoch+1}: validation F1 {metric:.4f}, "
                    f"best {state['best']:.4f}, {active_clock()-began:.1f} active s")
            if smoke_pause_after == epoch+1:
                return None,state
            if epoch+1 >= 12 and epoch+1-state["best_pass"] >= 5:
                break
    model.load_state_dict(state["best_weights"])
    params = sum(p.numel() for p in model.parameters())
    return model, {"train_patients":len(train),"validation_patients":len(validation),
                   "parameters":params,"best_validation_macro_f1":state["best"],"best_pass":state["best_pass"],
                   "passes":state["next_pass"],"this_session_seconds":active_clock()-start,"history":state["history"]}


def verify_smoke(frame: pd.DataFrame, split) -> None:
    train, val, _ = fold_frames(frame,split)
    train_subjects = sorted(train.subject_id.unique())[:8]
    val_subjects = sorted(val.subject_id.unique())[:4]
    selected = frame[frame.subject_id.isin(train_subjects+val_subjects)]
    recordings = prepare_recordings(selected,set(train_subjects))
    path = OUT/"smoke"
    pause = path/"PAUSE"
    first, state = fit("long",recordings,train_subjects,val_subjects,SEED,path/"resume",pause,
                       float("inf"),smoke_pause_after=1,max_passes=2)
    if first is not None or state["next_pass"] != 1:
        raise ValueError("Interruption check did not save the first pass")
    resumed, _ = fit("long",recordings,train_subjects,val_subjects,SEED,path/"resume",pause,float("inf"),max_passes=2)
    reference, _ = fit("long",recordings,train_subjects,val_subjects,SEED,path/"control",pause,float("inf"),max_passes=2)
    for key in resumed.state_dict():
        torch.testing.assert_close(resumed.state_dict()[key],reference.state_dict()[key],rtol=0,atol=0)
    truth = val[val.subject_id.isin(val_subjects)].iloc[:25]
    probs = score_fold(resumed,recordings,truth)
    np.testing.assert_allclose(probs.sum(axis=1),1,atol=1e-6)
    for name,schedule in CONTEXTS.items():
        network = FeatureContextNet(schedule).eval()
        radius = sum(schedule)
        x = torch.randn(1,100,90)
        recorded = torch.ones(1,100,dtype=torch.bool)
        altered = x.clone()
        altered[:, :50-radius] += 2
        altered[:, 51+radius:] += 2
        with torch.inference_mode():
            torch.testing.assert_close(network(x,recorded)[:,50],network(altered,recorded)[:,50],rtol=0,atol=0)
        log(f"{name} receptive field verified at {2*radius+1} epochs")
    atomic_json(path/"summary.json",{"passed":True,"scored_test_patients":0,
                                    "fit_patients":len(train_subjects),"validation_patients":len(val_subjects)})
    log("Training-only interruption and exact resume check passed")


def compare(frame: pd.DataFrame, candidate: str, reference: str) -> dict:
    truth, pred, ref = (frame[key].to_numpy() for key in (LABEL,candidate,reference))
    rows = [np.flatnonzero(frame.fold.to_numpy()==i) for i in range(5)]
    a,b = scores(frame,pred,rows),scores(frame,ref,rows)
    deltas = [float(f1_score(truth[i],pred[i],average="macro")-f1_score(truth[i],ref[i],average="macro"))
              for i in rows]
    ids = frame.subject_id.to_numpy()
    cm_a, patients = patient_confusions(ids,truth,pred)
    cm_b, _ = patient_confusions(ids,truth,ref)
    membership = frame.groupby("subject_id").fold.first().loc[patients].to_numpy()
    groups = [np.flatnonzero(membership==i) for i in range(5)]
    draws = cv_fold_draws(groups,2000,np.random.default_rng(SEED))
    ci = interval(cv_scores(cm_a,groups,draws)["macro_f1"]-cv_scores(cm_b,groups,draws)["macro_f1"])
    patient_values = []
    for subject, night in frame.groupby("subject_id",sort=False):
        y = night[LABEL]
        labels = sorted(y.unique())
        cmp = [f1_score(y,night[key],average="macro",labels=labels,zero_division=0)
               for key in (candidate,reference)]
        patient_values.append((subject,int(night.fold.iloc[0]),*cmp,cmp[0]-cmp[1]))
    columns = ["subject_id","fold","candidate_f1","reference_f1","delta"]
    pd.DataFrame(patient_values,columns=columns).to_csv(OUT/f"{candidate}_vs_{reference}_patients.csv",index=False)
    return {"candidate":a,"reference":b,"delta":float(np.mean(deltas)),"fold_deltas":deltas,"paired_ci":ci,
            "stage_deltas":{s:a[f"f1_{s}"]-b[f"f1_{s}"] for s in STAGES}}


def report(base: pd.DataFrame) -> None:
    keys = [*KEYS,LABEL,"fold"]
    aligned = base[keys+["prediction"]].rename(columns={"prediction":"baseline"})
    for variant in CONTEXTS:
        predictions = pd.concat([pd.read_parquet(OUT/f"fold-{f}"/variant/"predictions.parquet") for f in range(5)])
        predictions = base[keys].merge(predictions,on=keys,how="outer",validate="one_to_one",indicator=True)
        if len(predictions)!=len(base) or not predictions["_merge"].eq("both").all():
            raise ValueError("Saved prediction keys differ from the compact HGB")
        probabilities = base[keys].merge(predictions[keys+PROBABILITIES],on=keys,validate="one_to_one")[PROBABILITIES]
        columns = probabilities.to_numpy()
        aligned[variant] = columns.argmax(axis=1)
        aligned[f"{variant}_blend"] = (0.5*base[PROBABILITIES].to_numpy()+0.5*columns).argmax(axis=1)
        saved = base[keys].copy()
        saved[PROBABILITIES] = columns
        save_frame(OUT/f"{variant}_predictions.parquet",saved)
    save_frame(OUT/"paired_predictions.parquet",aligned)
    contrasts = {
        f"{a}_vs_{b}":compare(aligned,a,b)
        for a,b in [("short","baseline"),("long","baseline"),("long","short"),("short_blend","baseline"),
                    ("long_blend","baseline"),("long_blend","short_blend")]
    }
    context = contrasts["long_vs_short"]
    wins = lambda result: sum(d>0 for d in result["fold_deltas"])
    context["passes"] = context["delta"]>=0.005 and wins(context)>=4 and context["stage_deltas"]["N1"]>=-0.010
    for variant in ("long","long_blend"):
        c = contrasts[f"{variant}_vs_baseline"]
        c["passes"] = c["delta"]>=0.010 and wins(c)>=4 and min(c["stage_deltas"].values())>=-0.010
    errors = {variant:error_structure(aligned,variant) for variant in aligned.columns if variant not in keys}
    atomic_json(OUT/"summary.json",{"contrasts":contrasts,"errors":errors})
    for variant in ("short","long","short_blend","long_blend"):
        s = contrasts[f"{variant}_vs_baseline"]
        log(f'{variant}: {s["candidate"]["macro_f1"]:.4f} macro-F1, {s["delta"]:+.4f} over HGB18')


def execute(smoke: bool) -> int:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not (OUT/"FINDINGS.md").is_file():
        raise FileNotFoundError("Declare the experiments before fitting")
    frame,splits = development_frame()
    if smoke:
        verify_smoke(frame,splits[0])
        return 0
    if not (OUT/"smoke/summary.json").exists():
        raise ValueError("Run the training-only resume check first")
    base = pd.read_parquet(BASE)
    selections = [fold_frames(frame,split) for split in splits]
    expected = pd.concat([t.assign(fold=f)[[*KEYS,LABEL,"fold"]]
                          for f,(_,_,t) in enumerate(selections)])
    if len(base)!=len(expected) or len(expected.merge(base,on=[*KEYS,LABEL,"fold"],validate="one_to_one"))!=len(expected):
        raise ValueError("Model and archived comparator folds differ")
    src = Path(__file__).resolve()
    model_src = src.parents[1]/"src/physionet_sleep/experiments/feature_context.py"
    pin_manifest(OUT/"manifest.json",{"contexts":{k:list(v) for k,v in CONTEXTS.items()},"seed":SEED,
                                     "folds":[[sorted(part.subject_id.unique()) for part in split]
                                              for split in selections],
                                     "sources":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in [src,model_src,BASE]},
                                     "torch":torch.__version__})
    active = OUT/"active_budget.json"
    used = json.loads(active.read_text())["seconds"] if active.exists() else 0.0
    started = active_clock()
    deadline = started+86400-used
    try:
        for f,(train,val,test) in enumerate(selections):
            paths = {v:OUT/f"fold-{f}"/v for v in CONTEXTS}
            remaining = [v for v,path in paths.items() if not (path/"predictions.parquet").exists()]
            if not remaining:
                continue
            if (OUT/"PAUSE").exists() or active_clock() >= deadline:
                return 75
            ids = sorted(train.subject_id.unique())
            recordings = prepare_recordings(frame,set(ids))
            for name in remaining:
                path = paths[name]
                model,timing = fit(name,recordings,ids,sorted(val.subject_id.unique()),SEED+f,path,OUT/"PAUSE",deadline)
                if model is None:
                    log("Saved checkpoint and paused")
                    return 75
                probs = score_fold(model,recordings,test)
                atomic_json(path/"runtime.json",timing)
                write_predictions(path/"predictions.parquet",test,probs,f)
                log(f"fold {f+1}/5 {name} scored and saved")
        report(base)
    finally:
        atomic_json(active,{"seconds":used+active_clock()-started})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true",help="Verify resumability using training patients")
    with local_training_slot():
        raise SystemExit(execute(parser.parse_args().smoke))

