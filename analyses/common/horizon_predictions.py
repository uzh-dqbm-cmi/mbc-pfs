import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import CALIBRATION_HORIZONS as HORIZONS
from preprocess import preprocess_df
from analyses.common import bpc_data as B

OUT = Path("data/rebuttal/calibration/horizon_predictions.csv")
SITES = ("DFCI", "VICC")
LABELS = ["PATIENT_ID", "LINE", "PFS_TIME_DAYS", "PFS_EVENT"]
RISKS = [f"risk_{h}" for h in HORIZONS]


def fold_models():
    folds = json.loads(Path("results/gbsa/eval_metrics.json").read_text())["outer_folds"]
    for fold, info in sorted(folds.items()):
        split = pd.read_csv(info["split_path"])
        yield (int(fold[-1]), split.loc[split.split.ne("test"), "idx"].to_numpy(),
               split.loc[split.split.eq("test"), "idx"].to_numpy(),
               joblib.load(Path(info["results_path"]) / "model.joblib"))


def horizon_risks(model, x):
    positions = np.searchsorted(model.unique_times_, HORIZONS, side="right") - 1
    return 1 - model.predict_survival_function(x, return_array=True)[:, positions]


def predictions(msk, sites):
    oof, fold_ids = np.zeros((len(msk), len(HORIZONS))), np.full(len(msk), -1)
    external = {site: np.zeros((len(d), len(HORIZONS))) for site, d in sites.items()}
    folds = list(fold_models())
    for k, train, test, model in folds:
        oof[test] = horizon_risks(model, preprocess_df(train, test, test, msk)["X_test_np"])
        fold_ids[test] = k
        for site, d in sites.items():
            combined = pd.concat([msk, d.reindex(columns=msk.columns)], ignore_index=True)
            ext = np.arange(len(msk), len(combined))
            external[site] += horizon_risks(model, preprocess_df(train, ext, ext, combined)["X_test_np"]) / len(folds)
    return oof, fold_ids, external


def build():
    msk = pd.read_csv("data/design_matrix.csv")
    genie = pd.read_csv("data/genie/genie_design_matrix.csv")
    sites = {site: genie.loc[genie.PATIENT_ID.str.split("-").str[1].eq(site)].reset_index(drop=True)
             for site in SITES}
    oof, fold_ids, external = predictions(msk, sites)
    frames = [msk[LABELS].assign(cohort="MSK_OOF", outer_fold=fold_ids, **dict(zip(RISKS, oof.T)))]
    rows = B.mixed_endpoints(B.add_next_start(B.load_regimens()))
    for site, d in sites.items():
        frames.append(B.validation_labels(d[["PATIENT_ID", "LINE"]].assign(
            cohort=site, outer_fold=-1, **dict(zip(RISKS, external[site].T))), rows))
    return pd.concat(frames, ignore_index=True)[["cohort"] + LABELS + ["outer_fold"] + RISKS]


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build().to_csv(OUT, index=False)
