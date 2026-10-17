import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import build_pfs
from build_design_matrix import adjust_age
from config import MIN_PFS_DAYS, MSK_DATA_DIR, RADIOLOGY_REPORT_WINDOW_DAYS
from design_matrix_pipeline import DesignMatrixContext, assemble_design_matrix, build_all_modalities
from design_matrix_pipeline.modalities import diagnosis, treatment
from design_matrix_pipeline.utils import filter_and_sort_input
from preprocess import preprocess_df, typecast
from qc import qc_from_received_flags
from analyses.common.survival import load_curves

OUT = Path("data/rebuttal/eligibility")
WINDOW_DAYS = 180
CONTROL_ROWS = [0, 1700, 3400, 5100, 6800]
ABSENT_PREFIXES = ("PLANNED_", "RECEIVED_", "TUMOR_SITE_", "CLINICAL_GROUP_", "PATH_GROUP_", "HISTOLOGIC_", "SUMMARY_",
                   "CANCER_SITE_SUBSITE_")


def has_imaging(lines, days, window):
    return np.array([build_pfs.has_prior_measurement(days, pid, start, window)
                     for pid, start in zip(lines.PATIENT_ID, lines.LINE_START)], dtype=bool)


def added_endpoints(dropped, diag):
    lines = dropped.loc[dropped.LINE_SOURCE.eq(f"no_radiology_within_{RADIOLOGY_REPORT_WINDOW_DAYS}_prior")]
    patients = set(lines.PATIENT_ID)
    stage4 = set(diag.loc[diag.STAGE_CDM_DERIVED.eq("Stage 4"), "PATIENT_ID"])
    reports = build_pfs.after_metastasis(build_pfs.prep_progression(patients), stage4,
                                         build_pfs.first_metastasis_start_dates(patients))
    days = build_pfs.measurement_days(patients, progression_df=reports)
    assert not has_imaging(lines, days, RADIOLOGY_REPORT_WINDOW_DAYS).any()
    lines = lines.loc[has_imaging(lines, days, WINDOW_DAYS)]
    imaged = len(lines)
    lines = lines.loc[lines.ORIGINAL_PFS_EVENT.isin([0, 1]) & lines.ORIGINAL_PFS_TIME_DAYS.ge(MIN_PFS_DAYS)].copy()
    lines["PFS_EVENT"] = lines.ORIGINAL_PFS_EVENT.astype(int)
    lines["PFS_TIME_DAYS"] = lines.ORIGINAL_PFS_TIME_DAYS.astype("Float32")
    lines = build_pfs.apply_censor(lines)
    lines["LINE_SOURCE"] = "restored_180_day_eligibility"
    return lines, imaged


def added_features(endpoints, eligible, original, diag):
    top_agents = treatment.build_top_agent_lookup(treatment.load_treatment(MSK_DATA_DIR, eligible.PATIENT_ID.unique()))
    categories = diagnosis.top_categories(filter_and_sort_input(diag, set(eligible.PATIENT_ID)))
    controls = original.iloc[CONTROL_ROWS][["PATIENT_ID", "LINE"]]
    lines = pd.concat([endpoints, eligible.merge(controls, on=["PATIENT_ID", "LINE"], validate="one_to_one")],
                      ignore_index=True)
    ctx = DesignMatrixContext(data_dir=MSK_DATA_DIR, outcomes_path=io.StringIO(lines.to_csv(index=False)),
                              diag_path=Path("data/diagnosis_parsed.csv"), type_specific_patients=set(lines.PATIENT_ID),
                              patient_lot_info=lines[["PATIENT_ID", "LINE", "LINE_START", "EVENT_DAY"]],
                              top_agents=top_agents, diagnosis_categories=categories)
    features, _ = assemble_design_matrix(ctx, build_all_modalities(ctx))
    adjust_age(features)
    features = qc_from_received_flags(features)
    features["IS_MLOT1"] = features.LINE.eq(1).astype(int)
    absent = original.columns.difference(features.columns)
    assert absent.str.startswith(ABSENT_PREFIXES).all(), list(absent)
    features = features.reindex(columns=original.columns, fill_value=0)
    features = features.astype({c: "int8" for c in features.select_dtypes(include="bool")}).round(6)
    check = features.merge(controls, on=["PATIENT_ID", "LINE"]).set_index(["PATIENT_ID", "LINE"])
    expected = original.set_index(["PATIENT_ID", "LINE"]).loc[check.index]
    model_columns = typecast(original)[0].columns
    np.testing.assert_allclose(check[model_columns].to_numpy(float), expected[model_columns].to_numpy(float),
                               atol=1e-6, rtol=0, equal_nan=True)
    added = features.merge(endpoints[["PATIENT_ID", "LINE"]], on=["PATIENT_ID", "LINE"], validate="one_to_one")
    return added.loc[added.PFS_EVENT.ne(-1)].reset_index(drop=True)


def score(original, added):
    sf, grid = load_curves(original, "full_gbsa")
    folds = json.loads(Path("results/gbsa/eval_metrics.json").read_text())["outer_folds"]
    splits = {name: pd.read_csv(info["split_path"]) for name, info in folds.items()}
    held_out = {pid: name for name, split in splits.items()
                for pid in original.PATIENT_ID.iloc[split.loc[split.split.eq("test"), "idx"]]}
    names = list(folds)
    assigned = added.PATIENT_ID.map(
        lambda pid: held_out.get(pid, names[int(hashlib.sha256(pid.encode()).hexdigest(), 16) % len(names)]))
    combined = pd.concat([original, added], ignore_index=True)
    curves = np.full((len(added), len(grid)), np.nan)
    for name, info in folds.items():
        split = splits[name]
        train = split.loc[split.split.ne("test"), "idx"].to_numpy()
        test = split.loc[split.split.eq("test"), "idx"].to_numpy()
        selected = np.flatnonzero(assigned.eq(name))
        assert not set(added.PATIENT_ID.iloc[selected]) & set(original.PATIENT_ID.iloc[train])
        prep = preprocess_df(train, test, np.r_[test, len(original) + selected], combined)
        model = joblib.load(Path(info["results_path"]) / "model.joblib")
        columns = np.searchsorted(model.unique_times_, grid)
        assert np.array_equal(model.unique_times_[columns], grid)
        predicted = model.predict_survival_function(prep["X_test_np"], return_array=True)[:, columns]
        np.testing.assert_allclose(predicted[:len(test)], sf.iloc[test].to_numpy(), atol=1e-10, rtol=0)
        curves[selected] = predicted[len(test):]
    assert np.isfinite(curves).all()
    return grid, curves, len(set(added.PATIENT_ID) - set(held_out))


def main():
    original = pd.read_csv("data/design_matrix.csv")
    eligible = pd.read_csv("data/BREAST_pfs.csv")
    diag = pd.read_csv("data/diagnosis_parsed.csv")
    endpoints, imaged = added_endpoints(pd.read_csv("data/BREAST_dropped_pfs_lines.csv"), diag)
    added = added_features(endpoints, eligible, original, diag)
    grid, curves, new_patients = score(original, added)
    OUT.mkdir(parents=True, exist_ok=True)
    added.to_csv(OUT / "added_180_day_features.csv", index=False)
    np.savez_compressed(OUT / "added_180_day_survival.npz", time=grid, surv=curves)
    print(f"Imaging in prior {WINDOW_DAYS} days: {imaged} lines; usable endpoints: {len(endpoints)} lines from "
          f"{endpoints.PATIENT_ID.nunique()} patients; QC excluded: {len(endpoints) - len(added)}; retained: {len(added)} lines "
          f"from {added.PATIENT_ID.nunique()} patients ({new_patients} not in the original cohort).", flush=True)


if __name__ == "__main__":
    main()
