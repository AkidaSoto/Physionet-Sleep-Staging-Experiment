
"""Compare observed feature measurements on matched repairs and persistent errors."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPLORATION = ROOT/"artifacts/exploration"
OUT = EXPLORATION/"cinc-probability-diagnostic"
LABEL = "label.stage_30s"
KEYS = ["subject_id","epoch_index",LABEL,"fold"]
PAIRS = [(2,3),(3,2),(2,1),(1,2),(1,0),(0,1),(4,1),(1,4)]
STAGES = ["Wake","N1","N2","N3","REM"]
FEATURES = [
    "c_rel_theta","c_rel_alpha","c_rel_beta","c_log_power","o_rel_alpha","eog_high",
    "aperiodic_exponent","delta_power","rel_theta","rel_beta","eye_movement_activity",
    "eye_movement_baseline","eog_rem_vs_noise_db","eog_rem_vs_sem_db","emg_floor_db",
    "frontal_slow_wave_fraction","occipital_rel_alpha","burst_seconds","frontal_delta_control",
]


def placement(sample: np.ndarray, reference: np.ndarray) -> np.ndarray:
    result = np.full(sample.shape[1],np.nan)
    for k,(values,rivals) in enumerate(zip(sample.T,reference.T)):
        a,b = values[np.isfinite(values)],np.sort(rivals[np.isfinite(rivals)])
        if len(a) and len(b):
            result[k] = (np.searchsorted(b,a,side="left")+np.searchsorted(b,a,side="right")).mean()/(2*len(b))
    return result


def direction(frame,pair):
    true,rival = pair
    mistakes = frame[frame[LABEL].eq(true)&frame.tree.eq(rival)]
    references = frame.loc[frame[LABEL].eq(rival)&frame.tree.eq(rival)]
    keys = ["subject_id","quarter","ref_position"]
    ref_groups = {key:g[FEATURES].to_numpy() for key,g in references.groupby(keys,sort=False) if len(g)>=3}
    entries = []
    for key,group in mistakes.groupby(keys+["margin_bin"],observed=True,sort=False):
        ref = ref_groups.get(key[:3])
        if ref is None:
            continue
        recover = group.loc[group.outcome.eq("repaired"),FEATURES].to_numpy()
        remain = group.loc[group.outcome.eq("persistent"),FEATURES].to_numpy()
        if not len(recover) or not len(remain):
            continue
        a,b = placement(recover,ref),placement(remain,ref)
        entries.extend((key[0],group.fold.iloc[0],key[2],feature,ar,br,len(recover),len(remain),min(len(recover),len(remain)))
                       for feature,ar,br in zip(FEATURES,a,b) if np.isfinite(ar) and np.isfinite(br))
    columns = ["subject","fold","position","feature","recovered_auc","persistent_auc","n_recovered","n_persistent","w"]
    return pd.DataFrame(entries,columns=columns)


def main():
    f = pd.read_parquet(EXPLORATION/"cinc-cap-audit/feature_failure_epochs.parquet",
                        columns=[*KEYS,"ref_position",*FEATURES])
    cases = pd.read_parquet(OUT/"error_cases.parquet",columns=[*KEYS,"tree","outcome","quarter","margin"])
    z = f.merge(cases,on=KEYS,validate="one_to_one")
    if len(z)!=len(f) or len(z)!=len(cases):
        raise ValueError("Archived measurements and error cases have different keys")
    z["margin_bin"] = pd.cut(z.margin,[-1e-6,.1,.25,.5,1.00001])
    results = []
    for pair in PAIRS:
        for (feature,),group in direction(z,pair).groupby(["feature"]):
            r,p = [np.average(group[k],weights=group.w) for k in ("recovered_auc","persistent_auc")]
            results.append({"truth":STAGES[pair[0]],"hgb":STAGES[pair[1]],"feature":feature,
                            "recovered_auc_high_vs_rival":r,"persistent_auc_high_vs_rival":p,
                            "rank_delta":r-p,"patients":group.subject.nunique(),
                            "matched_recoveries":group.n_recovered.sum(),"matched_persistent":group.n_persistent.sum()})
    pd.DataFrame(results).to_csv(OUT/"matched_feature_measurements.csv",index=False,float_format="%.6f")


if __name__=="__main__":
    main()

