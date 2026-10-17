import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv

from preprocess import preprocess_df
from analyses.common.survival import ibs_evaluator, load_curves

OUT = Path("results/rebuttal/eligibility")


def excluded_lines():
    original = pd.read_csv("data/design_matrix.csv")
    labels = pd.read_csv("data/BREAST_pfs.csv")
    added = pd.read_csv("data/dropped_lines.csv")
    added = added.drop(columns=["PFS_TIME_DAYS", "PFS_EVENT"]).merge(
        labels[["PATIENT_ID", "LINE", "PFS_TIME_DAYS", "PFS_EVENT"]],
        on=["PATIENT_ID", "LINE"], validate="one_to_one")
    added["IS_MLOT1"] = added.LINE.eq(1).astype(int)
    combined = pd.concat([original, added[original.columns]], ignore_index=True)
    assert len(combined) == len(labels) and not combined.duplicated(["PATIENT_ID", "LINE"]).any()
    return original, added, combined


def outer_folds(added):
    folds = json.loads(Path("results/gbsa/eval_metrics.json").read_text())["outer_folds"]
    splits = {name: pd.read_csv(info["split_path"]) for name, info in folds.items()}
    patient_fold = {pid: name for name, split in splits.items()
                    for pid in split.loc[split.split.eq("test"), "PATIENT_ID"].unique()}
    new_ids = sorted(set(added.PATIENT_ID) - set(patient_fold))
    order = np.random.default_rng(42).permutation(len(new_ids))
    names = list(folds)
    patient_fold.update({new_ids[j]: names[i % len(names)] for i, j in enumerate(order)})
    return folds, splits, added.PATIENT_ID.map(patient_fold), new_ids


def score_excluded(original, combined, folds, splits, assigned):
    sf, grid = load_curves(original, "full_gbsa")
    added_curves = np.full((len(combined) - len(original), len(grid)), np.nan)
    for name, info in folds.items():
        split = splits[name]
        train = split.loc[split.split.ne("test"), "idx"].to_numpy()
        test = split.loc[split.split.eq("test"), "idx"].to_numpy()
        selected = np.flatnonzero(assigned.eq(name))
        targets = np.r_[test, len(original) + selected]
        assert not set(combined.iloc[targets].PATIENT_ID) & set(original.iloc[train].PATIENT_ID)
        prep = preprocess_df(train, test, targets, combined)
        model = joblib.load(Path(info["results_path"]) / "model.joblib")
        time_cols = np.searchsorted(model.unique_times_, grid)
        np.testing.assert_array_equal(model.unique_times_[time_cols], grid)
        predicted = model.predict_survival_function(prep["X_test_np"], return_array=True)[:, time_cols]
        np.testing.assert_allclose(predicted[:len(test)], sf.iloc[test].to_numpy(), rtol=1e-6, atol=1e-8)
        added_curves[selected] = predicted[len(test):]
        print(f"{name}: scored {len(selected):,} QC-excluded lines; saved test curves matched.", flush=True)
    assert np.isfinite(added_curves).all()
    return sf, grid, added_curves


def main():
    original, added, combined = excluded_lines()
    folds, splits, assigned, new_ids = outer_folds(added)
    sf, grid, added_curves = score_excluded(original, combined, folds, splits, assigned)
    curves = np.vstack([sf.to_numpy(), added_curves])
    times = np.unique(original.loc[original.PFS_EVENT.eq(0), "PFS_TIME_DAYS"])
    rows = []
    for name, idx in [("Original retained cohort", np.arange(len(original))),
                      ("All eligible lines, including QC exclusions", np.arange(len(combined))),
                      ("QC-excluded lines only", np.arange(len(original), len(combined)))]:
        frame, survival = combined.iloc[idx], curves[idx]
        ev = EvalSurv(pd.DataFrame(survival.T, index=grid),
                      frame.PFS_TIME_DAYS.to_numpy(), frame.PFS_EVENT.to_numpy())
        bs = ibs_evaluator(survival, grid, frame).brier_score_multiple_points(times)
        rows.append(dict(Cohort=name, Lines=len(frame), Patients=frame.PATIENT_ID.nunique(),
                         C=float(ev.concordance_td("adj_antolini")),
                         IBS=float(np.trapezoid(bs, times) / (times[-1] - times[0]))))
    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "qc_sensitivity.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
