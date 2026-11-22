
"""Read the additive night-history effects back on the outer evaluation patients."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_staging_probability_diagnostic import OUT, FULL_KEYS, LABEL, PHYSIOLOGY, PROBABILITIES, STAGES, night_history

GROUPS = {
    "slow_history":[0,1],
    "low_chin_tone":[2],
    "eye_chin_anchor":[3],
}

def main():
    p = pd.read_parquet(OUT/"physiology_predictions.parquet")
    errors = pd.read_parquet(OUT/"error_cases.parquet",columns=[*FULL_KEYS,"interior"])
    p = p.merge(errors,on=FULL_KEYS,validate="one_to_one")
    assert len(p)==len(errors)==288923
    x = np.empty((len(p),4),dtype=np.float32)
    for subject,rows in p.groupby("subject_id",sort=False).indices.items():
        x[rows] = night_history(subject)[p.iloc[rows].epoch_index.to_numpy(),-4:]
    items = []
    coeff = []
    for fold in range(5):
        loc = p.fold.eq(fold).to_numpy()
        params = np.load(OUT/f"fold-{fold}/physiology_coefficients.npz")
        assert params["columns"][-4:].tolist()==PHYSIOLOGY
        w = params["weights"][-5:-1,:]
        xx = np.clip((x[loc]-params["means"][-4:])/params["scales"][-4:],-10,10)
        logp = np.log(p.loc[loc,PROBABILITIES].to_numpy().clip(1e-12))
        y = p.loc[loc,LABEL].to_numpy()
        inside = p.loc[loc,"interior"].to_numpy()
        for (a,b) in [(3,2),(4,1),(4,2)]:
            idx = ((y==a)|(y==b))&inside
            full = roc_auc_score(y[idx]==a,logp[idx,a]-logp[idx,b])
            for i,feature in enumerate(PHYSIOLOGY):
                coeff.append({"fold":fold,"true_stage":STAGES[a],"rival":STAGES[b],"feature":feature,
                              "log_odds_per_training_sd":float(w[i,a]-w[i,b])})
            for key,cols in GROUPS.items():
                ablated = logp-xx[:,cols]@w[cols,:]
                auc = roc_auc_score(y[idx]==a,ablated[idx,a]-ablated[idx,b])
                q = softmax(ablated,axis=1)[np.arange(len(y)),y]
                items.append({"fold":fold,"true_stage":STAGES[a],"rival":STAGES[b],"group":key,
                              "interior_auc":full,"without_group_auc":auc,"auc_contribution":float(full-auc),
                              "pooled_fold_log_loss_without_group":float(-np.log(q.clip(1e-12)).mean())})
    pd.DataFrame(items).to_csv(OUT/"night_feature_contributions.csv",index=False)
    pd.DataFrame(coeff).to_csv(OUT/"night_feature_coefficients.csv",index=False)
    print(pd.DataFrame(items).groupby(["true_stage","rival","group"]).auc_contribution.agg(["mean","min","max"]).round(4))

if __name__=="__main__":
    main()

