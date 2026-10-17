import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import monotonic

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import joblib
import numpy as np
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis

from preprocess import preprocess_df
from analyses.cohort_selection.qc_sensitivity import excluded_lines, outer_folds, score_excluded

OUT = Path("results/rebuttal/qc_refit")
CLASSIFICATION = Path("results/rebuttal/eligibility/late_start/line_classification.csv")
IDENTITY = ["PATIENT_ID", "LINE", "LINE_START", "PFS_TIME_DAYS", "PFS_EVENT"]
SIX = ["IS_MLOT1", "HR", "HER2", "ECOG_LAST_OBS", "TUMOR_SITE_LIVER_IN_WINDOW", "CA15_3_LAST_OBS_LOG"]
META_COLS = ["CATEGORY", "LATE_START", "QC_CHEMO_ENDO", "QC_HRNEG", "ADD_DAY", "START_CLASS", "SUBTYPE", "FIRST_LINE"]
GNRH_CATEGORIES = ("Co-prescription: GnRH agent only (no other endocrine agent in the line)",
                   "Co-prescription: GnRH agent at start, non-GnRH endocrine added after day 28")
VARIANTS = ["baseline", "late525", "late525_censor", "restated_rule", "chemo_endo_all", "qc_all"]


def variant_mask(name, meta):
    not_hrneg = ~meta.QC_HRNEG
    if name == "baseline":
        return np.zeros(len(meta), bool)
    if name in ("late525", "late525_censor"):
        return (meta.LATE_START & not_hrneg).to_numpy()
    if name == "restated_rule":
        return ((meta.LATE_START | meta.CATEGORY.isin(GNRH_CATEGORIES)) & not_hrneg).to_numpy()
    if name == "chemo_endo_all":
        return (meta.QC_CHEMO_ENDO & not_hrneg).to_numpy()
    if name == "qc_all":
        return np.ones(len(meta), bool)
    raise ValueError(name)


def restated_rule_note():
    original, added, meta, _, _ = load_inputs()
    late, restated = variant_mask("late525", meta), variant_mask("restated_rule", meta)
    gnrh = restated & ~late
    return (f"Restated-rule variant: {len(original) + restated.sum():,} lines = {len(original):,} retained + {late.sum():,} "
            f"late-start + {gnrh.sum():,} GnRH-group lines ({added.HR[gnrh].eq(1).sum():,} HR-positive).")


def load_inputs():
    original, added, combined = excluded_lines()
    classes = pd.read_csv(CLASSIFICATION)
    meta = added[["PATIENT_ID", "LINE"]].merge(classes[["PATIENT_ID", "LINE"] + META_COLS], on=["PATIENT_ID", "LINE"],
                                               how="left", validate="one_to_one")
    assert meta.CATEGORY.notna().all() and len(combined) == 11085
    folds, splits, meta["FOLD"], new_ids = outer_folds(added)
    meta["NEW_PATIENT"] = added.PATIENT_ID.isin(new_ids)
    return original, added[original.columns].reset_index(drop=True), meta, folds, splits


def fold_rows(split, info, fold):
    restored = np.flatnonzero(info.SOURCE.eq("restored"))
    held_out = info.FOLD.to_numpy()[restored] == fold
    return (np.r_[split.loc[split.split.ne("test"), "idx"].to_numpy(), restored[~held_out]],
            np.r_[split.loc[split.split.eq("test"), "idx"].to_numpy(), restored[held_out]])


def build_cohort(name, original, added, meta):
    sel = variant_mask(name, meta)
    restored, info = added[sel].copy(), meta[sel].copy()
    if name == "late525_censor":
        add_day = info.ADD_DAY.to_numpy(float)
        t = restored.PFS_TIME_DAYS.to_numpy(float)
        censor = add_day < t
        restored.loc[censor, "PFS_TIME_DAYS"] = add_day[censor]
        restored.loc[censor, "PFS_EVENT"] = 0
        assert restored.PFS_TIME_DAYS.gt(28).all()
    features = pd.concat([original, restored], ignore_index=True)
    base = pd.DataFrame({"SOURCE": "retained", "FOLD": None, "NEW_PATIENT": False}, index=range(len(original)))
    for c in META_COLS:
        base[c] = np.nan
    info = info.assign(SOURCE="restored").reset_index(drop=True)
    info = pd.concat([base, info[base.columns]], ignore_index=True)
    info = pd.concat([features[IDENTITY].reset_index(drop=True), info], axis=1)
    return features, info


def fit_one(name, fold, force=False):
    started = monotonic()
    original, added, meta, folds, splits = load_inputs()
    features, info = build_cohort(name, original, added, meta)
    out = OUT / name / fold
    out.mkdir(parents=True, exist_ok=True)
    if (out / "surv_test.npz").exists() and (out / "surv_test_reference.npz").exists() and not force:
        return f"{name}/{fold}: reused."
    train, test = fold_rows(splits[fold], info, fold)
    assert not set(features.PATIENT_ID.iloc[train]) & set(features.PATIENT_ID.iloc[test])
    prep = preprocess_df(train, test, test, features)
    saved = joblib.load(Path(folds[fold]["results_path"]) / "model.joblib")
    model = GradientBoostingSurvivalAnalysis(**saved.get_params())
    model.fit(prep["X_train_np"], prep["y_train_sksurv"])
    curves = model.predict_survival_function(prep["X_test_np"], return_array=True)
    keep = model.unique_times_ < 730
    note = ""
    if name == "baseline":
        with np.load(Path(folds[fold]["results_path"]) / "surv_test.npz") as s:
            same_grid = np.array_equal(s["time"], model.unique_times_[keep]) and np.array_equal(s["idx_test"], test)
            diff = np.abs(curves[:, keep] - s["surv"]).max() if same_grid else np.inf
        note = f" baseline check: grid {'equal' if same_grid else 'DIFFERENT'}, max |diff| vs saved curves {diff:.2e}."
    np.savez_compressed(out / "surv_test.npz", time=model.unique_times_[keep], surv=curves[:, keep], idx_test=test)
    joblib.dump(model, out / "model.joblib")
    names = prep["numeric_cols"] + prep["binary_cols"]
    json.dump(sorted(zip(names, map(float, model.feature_importances_)), key=lambda kv: -kv[1]),
              open(out / "importance.json", "w"), indent=1)
    frame = features[IDENTITY + SIX]
    prep_ref = preprocess_df(train, test, test, frame)
    cox = CoxPHSurvivalAnalysis(alpha=1.0, ties="efron", n_iter=1000).fit(prep_ref["X_train_np"], prep_ref["y_train_sksurv"])
    ref = cox.predict_survival_function(prep_ref["X_test_np"], return_array=True)
    keep_ref = cox.unique_times_ < 730
    np.savez_compressed(out / "surv_test_reference.npz", time=cox.unique_times_[keep_ref], surv=ref[:, keep_ref], idx_test=test)
    return (f"{name}/{fold}: trained on {len(train):,} lines ({(train >= len(original)).sum():,} restored), tested on {len(test):,} "
            f"({(test >= len(original)).sum():,} restored); {monotonic() - started:.0f} s.{note}")


def score_fixed_original(force=False):
    out = OUT / "fixed_original"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "curves.npz").exists() and not force:
        return "fixed_original: reused."
    original, added, meta, folds, splits = load_inputs()
    _, grid, curves = score_excluded(original, pd.concat([original, added], ignore_index=True), folds, splits, meta.FOLD)
    np.savez_compressed(out / "curves.npz", time=grid, surv=curves)
    pd.concat([added[IDENTITY], meta.drop(columns=["PATIENT_ID", "LINE"])], axis=1).to_csv(out / "added_meta.csv", index=False)
    return f"fixed_original: scored {len(added):,} excluded lines with the saved models; saved test curves matched."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    original, added, meta, folds, _ = load_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in variants:
        features, info = build_cohort(name, original, added, meta)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        info.to_csv(OUT / name / "cohort.csv", index=False)
        print(f"{name}: {len(features):,} lines, {features.PATIENT_ID.nunique():,} patients, "
              f"{int(info.SOURCE.eq('restored').sum()):,} restored, events {features.PFS_EVENT.mean():.3f}.", flush=True)
    print(score_fixed_original(args.force), flush=True)
    jobs = [(name, fold) for name in variants for fold in folds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fit_one, name, fold, args.force): (name, fold) for name, fold in jobs}
        for future in as_completed(futures):
            print(future.result(), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
