import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_variable, "2")

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import KaplanMeierFitter
from pycox.evaluation import EvalSurv
from scipy.special import expit, logit
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from SurvivalEVAL.Evaluator import SurvivalEvaluator
from tabulate import tabulate

from config import EVAL_HORIZONS
from preprocess import preprocess_df
from analyses.common.auc_bootstrap import bootstrap_auc
from analyses.common.ibs_bootstrap import bootstrap_ibs
from analyses.common.survival import patient_bootstrap, patient_draws
from analyses.calibration import FROZEN, HORIZONS, analyze_horizon, cell, ipcw
from analyses.calibration import OUT as CALIBRATION
from utils import compute_time_dependent_auc

OUT = Path("results/rebuttal/external")
IDENTITY = ["PATIENT_ID", "LINE", "LINE_START", "PFS_TIME_DAYS", "PFS_EVENT"]
SIX = ["IS_MLOT1", "HR", "HER2", "ECOG_LAST_OBS", "TUMOR_SITE_LIVER_IN_WINDOW", "CA15_3_LAST_OBS_LOG"]
SITES = ("DFCI", "VICC")
ENDPOINTS = {"mixed": "BPC PFS-I + Mixed", "pfs_i": "BPC PFS-I"}
MODELS = {"full_gbsa": "Full GBSA", "reference": "Six-variable Cox reference"}
METRICS = ["C", "IBS", "mean_auc", "auc_at_1y", "auc_at_2y"]
ED17_METRICS = {"C": "C", "mean_auc": "AUC", "auc_at_1y": "AUC@1y", "auc_at_2y": "AUC@2y", "IBS": "IBS"}
N_BOOT = 1000
CV_FOLDS = 5
VARIANTS = {
    "original": None,
    "intercept, apparent": dict(slope=False),
    "intercept and slope, apparent": dict(slope=True),
    "intercept, 5-fold CV": dict(slope=False, folds=CV_FOLDS),
    "intercept and slope, 5-fold CV": dict(slope=True, folds=CV_FOLDS),
    "intercept and slope, fit at other site": dict(slope=True, transfer=True),
}


def load_msk():
    return pd.read_csv("data/design_matrix.csv")


def external_cohorts():
    genie = pd.read_csv("data/genie/genie_design_matrix.csv")
    genie["site"] = genie.PATIENT_ID.str.split("-").str[1]
    frozen = pd.read_csv(FROZEN)
    cohorts = {}
    for site in SITES:
        sub = genie.loc[genie.site.eq(site)].drop(columns="site").reset_index(drop=True)
        cohorts[(site, "pfs_i")] = sub
        labels = frozen.loc[frozen.cohort.eq(site), ["PATIENT_ID", "LINE", "PFS_TIME_DAYS", "PFS_EVENT"]]
        mixed = sub.drop(columns=["PFS_TIME_DAYS", "PFS_EVENT"]).merge(labels, on=["PATIENT_ID", "LINE"], how="inner")
        if len(mixed) != len(labels):
            raise ValueError(f"{site}: {len(mixed)} merged rows for {len(labels)} frozen rows.")
        cohorts[(site, "mixed")] = mixed.reset_index(drop=True)
    return cohorts


def availability(msk, cohorts):
    rows = []
    for label, frame in [("MSK-CHORD", msk)] + [(s, cohorts[(s, "mixed")]) for s in SITES]:
        rows.append(dict(
            cohort=label, lines=len(frame), patients=frame.PATIENT_ID.nunique(),
            first_line_pct=100 * frame.IS_MLOT1.mean(), HR_pos_pct=100 * frame.HR.mean(),
            HER2_pos_pct=100 * frame.HER2.mean(),
            ECOG_recorded_pct=100 * (1 - frame.ECOG_MISSING).mean(),
            ECOG_ge1_pct=100 * (frame.ECOG_LAST_OBS >= 1).mean(),
            liver_in_window_pct=100 * frame.TUMOR_SITE_LIVER_IN_WINDOW.mean(),
            CA15_3_available_pct=100 * frame.CA15_3_LAST_OBS_LOG.notna().mean()))
    return pd.DataFrame(rows)


def censoring(msk, cohorts):
    rows = []
    for label, endpoint, frame in [("MSK-CHORD", "mLoT PFS", msk)] + [
            (site, ENDPOINTS[endpoint], cohorts[(site, endpoint)]) for endpoint in ENDPOINTS for site in SITES]:
        t, event = frame.PFS_TIME_DAYS.to_numpy(float), frame.PFS_EVENT.to_numpy(int)
        km = KaplanMeierFitter().fit(t, 1 - event)
        rows.append(dict(cohort=label, endpoint=endpoint, lines=len(frame), censored_pct=100 * (1 - event).mean(),
                         uncensored_180=float(km.predict(180.0)), uncensored_365=float(km.predict(365.0))))
    return pd.DataFrame(rows)


def fold_info():
    folds = json.loads(Path("results/gbsa/eval_metrics.json").read_text())["outer_folds"]
    out = []
    for fold, info in sorted(folds.items()):
        split = pd.read_csv(info["split_path"])
        train = split.loc[split.split.ne("test"), "idx"].to_numpy()
        out.append((fold, train, Path(info["results_path"]) / "model.joblib"))
    return out


def score_fold(msk, cohort, train, model_path):
    combined = pd.concat([msk, cohort.reindex(columns=msk.columns)], ignore_index=True)
    ext = np.arange(len(msk), len(combined))
    prep = preprocess_df(train, ext, ext, combined)
    gbsa = joblib.load(model_path)
    if gbsa.n_features_in_ != prep["X_test_np"].shape[1]:
        raise ValueError("The GBSA fold model and the external feature view differ in width.")
    curves = {"full_gbsa": _trim(gbsa.predict_survival_function(prep["X_test_np"], return_array=True),
                                 np.asarray(gbsa.unique_times_, float))}
    frame = pd.concat([msk[IDENTITY + SIX], cohort[IDENTITY + SIX]], ignore_index=True)
    prep_ref = preprocess_df(train, ext, ext, frame)
    cox = CoxPHSurvivalAnalysis(alpha=1.0, ties="efron", n_iter=1000)
    cox.fit(prep_ref["X_train_np"], prep_ref["y_train_sksurv"])
    curves["reference"] = _trim(cox.predict_survival_function(prep_ref["X_test_np"], return_array=True),
                                np.asarray(cox.unique_times_, float))
    coefficients = dict(zip(prep_ref["feature_cols"], map(float, cox.coef_)))
    return curves, coefficients, prep["y_train_sksurv"]


def _trim(curves, grid):
    keep = grid <= 730.0
    return curves[:, keep], grid[keep]


def risk_at(curves, grid, horizon):
    idx = max(0, int(np.searchsorted(grid, horizon, side="right") - 1))
    return 1.0 - curves[:, idx]


def point_metrics(curves, grid, dm, y_train):
    t, e = dm.PFS_TIME_DAYS.to_numpy(float), dm.PFS_EVENT.to_numpy(int)
    c = float(EvalSurv(pd.DataFrame(curves.T, index=grid), t, e).concordance_td("adj_antolini"))
    evaluator = SurvivalEvaluator(pred_survs=curves, time_coordinates=grid, test_event_times=t,
                                  test_event_indicators=e, train_event_times=y_train["time"],
                                  train_event_indicators=y_train["event"].astype(int))
    auc = compute_time_dependent_auc(curves, grid, y_train.copy(),
                                     Surv.from_arrays(e.astype(bool), t), EVAL_HORIZONS)
    own = SurvivalEvaluator(pred_survs=curves, time_coordinates=grid, test_event_times=t,
                            test_event_indicators=e, train_event_times=t, train_event_indicators=e)
    return dict(C=c, IBS_train_ipcw=float(evaluator.integrated_brier_score()),
                IBS_test_ipcw=float(own.integrated_brier_score()),
                mean_auc_train_ipcw=float(auc["mean_auc"]))


def discrimination(msk, cohorts, force=False):
    by_fold_path, summary_path = OUT / "discrimination_by_fold.csv", OUT / "discrimination_summary.csv"
    if by_fold_path.exists() and summary_path.exists() and not force:
        print("Reuse saved discrimination results.", flush=True)
        return
    folds = fold_info()
    by_fold, samples, coefficients, horizon_risks = [], {}, [], []
    for (site, endpoint), cohort in cohorts.items():
        dm = cohort[["PATIENT_ID", "PFS_TIME_DAYS", "PFS_EVENT"]].reset_index(drop=True)
        _, _, weights = patient_draws(dm, N_BOOT, 42)
        den = None
        started = monotonic()
        for fold, train, model_path in folds:
            curves, coef, y_train = score_fold(msk, cohort, train, model_path)
            coefficients.append(dict(fold=fold, site=site, endpoint=endpoint, **coef))
            for model, (surv, grid) in curves.items():
                point = point_metrics(surv, grid, dm, y_train)
                conc = patient_bootstrap(surv, grid, dm, weights=weights, den=den)
                den = conc["den"]
                ibs_point, ibs_samples = bootstrap_ibs(surv, grid, dm, weights)
                auc_point, auc_samples = bootstrap_auc(surv, grid, dm, weights)
                by_fold.append(dict(site=site, endpoint=endpoint, fold=fold, model=model,
                                    n_lines=len(dm), n_patients=dm.PATIENT_ID.nunique(),
                                    events=int(dm.PFS_EVENT.sum()), C=conc["C"], C_evalsurv=point["C"],
                                    IBS=ibs_point, IBS_train_ipcw=point["IBS_train_ipcw"],
                                    IBS_test_ipcw=point["IBS_test_ipcw"],
                                    mean_auc=auc_point["mean_auc"], auc_at_1y=auc_point["auc_at_1y"],
                                    auc_at_2y=auc_point["auc_at_2y"],
                                    mean_auc_train_ipcw=point["mean_auc_train_ipcw"]))
                samples[(site, endpoint, model, fold)] = pd.DataFrame(dict(
                    C=conc["samples"], IBS=ibs_samples, mean_auc=auc_samples[:, 0],
                    auc_at_1y=auc_samples[:, 1], auc_at_2y=auc_samples[:, 2]))
                risks = {f"risk_{h}": risk_at(surv, grid, h) for h in HORIZONS}
                horizon_risks.append(pd.DataFrame(dict(site=site, endpoint=endpoint, fold=fold, model=model,
                                                       PATIENT_ID=dm.PATIENT_ID, LINE=cohort.LINE, **risks)))
                print(f"{site} {endpoint} {fold} {model}: C {conc['C']:.4f}, IBS {ibs_point:.4f}, "
                      f"mean AUC {auc_point['mean_auc']:.4f}; {monotonic() - started:.0f} s.", flush=True)
    by_fold = pd.DataFrame(by_fold)
    OUT.mkdir(parents=True, exist_ok=True)
    by_fold.to_csv(by_fold_path, index=False)
    pd.DataFrame(coefficients).to_csv(OUT / "reference_coefficients.csv", index=False)
    pd.concat(horizon_risks, ignore_index=True).to_csv(OUT / "external_fold_horizon_risks.csv", index=False)
    summary, sample_rows = [], []
    for (site, endpoint) in cohorts:
        fold_names = [fold for fold, _, _ in folds]
        mean_samples = {}
        for model in MODELS:
            stack = np.stack([samples[(site, endpoint, model, fold)][METRICS].to_numpy() for fold in fold_names])
            mean_samples[model] = stack.mean(axis=0)
            points = by_fold.loc[by_fold.site.eq(site) & by_fold.endpoint.eq(endpoint) & by_fold.model.eq(model)]
            for j, metric in enumerate(METRICS):
                low, high = np.quantile(mean_samples[model][:, j], [.025, .975])
                summary.append(dict(site=site, endpoint=endpoint, model=model, metric=metric,
                                    point=points[metric].mean(), fold_sd=points[metric].std(ddof=0),
                                    CI_low=low, CI_high=high, fold_min=points[metric].min(),
                                    fold_max=points[metric].max()))
            for metric in ("IBS_train_ipcw", "mean_auc_train_ipcw"):
                summary.append(dict(site=site, endpoint=endpoint, model=model, metric=metric,
                                    point=points[metric].mean(), fold_sd=points[metric].std(ddof=0),
                                    CI_low=np.nan, CI_high=np.nan, fold_min=points[metric].min(),
                                    fold_max=points[metric].max()))
            frame = pd.DataFrame(mean_samples[model], columns=METRICS)
            frame.insert(0, "draw", np.arange(N_BOOT))
            frame.insert(0, "model", model)
            frame.insert(0, "endpoint", endpoint)
            frame.insert(0, "site", site)
            sample_rows.append(frame)
        delta = mean_samples["full_gbsa"] - mean_samples["reference"]
        full = by_fold.loc[by_fold.site.eq(site) & by_fold.endpoint.eq(endpoint) & by_fold.model.eq("full_gbsa")].set_index("fold")
        ref = by_fold.loc[by_fold.site.eq(site) & by_fold.endpoint.eq(endpoint) & by_fold.model.eq("reference")].set_index("fold")
        for j, metric in enumerate(METRICS):
            fold_delta = (full[metric] - ref[metric]).loc[fold_names]
            low, high = np.quantile(delta[:, j], [.025, .975])
            summary.append(dict(site=site, endpoint=endpoint, model="full_gbsa minus reference", metric=metric,
                                point=fold_delta.mean(), fold_sd=fold_delta.std(ddof=0), CI_low=low, CI_high=high,
                                fold_min=fold_delta.min(), fold_max=fold_delta.max(),
                                folds_favoring_full=int((fold_delta > 0).sum()) if metric != "IBS" else int((fold_delta < 0).sum()),
                                bootstrap_share_favoring_full=float((delta[:, j] > 0).mean()) if metric != "IBS" else float((delta[:, j] < 0).mean())))
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    pd.concat(sample_rows, ignore_index=True).to_csv(OUT / "discrimination_samples.csv", index=False)


def check_frozen_match():
    risks = pd.read_csv(OUT / "external_fold_horizon_risks.csv")
    frozen = pd.read_csv(FROZEN)
    sub = risks.loc[risks.endpoint.eq("mixed") & risks.model.eq("full_gbsa")]
    mean = sub.groupby(["site", "PATIENT_ID", "LINE"], as_index=False)[[f"risk_{h}" for h in HORIZONS]].mean()
    merged = mean.merge(frozen, left_on=["site", "PATIENT_ID", "LINE"], right_on=["cohort", "PATIENT_ID", "LINE"])
    worst = max(np.abs(merged[f"risk_{h}_x"] - merged[f"risk_{h}_y"]).max() for h in HORIZONS)
    print(f"Frozen-probability replay: {len(merged)} regimens matched; max abs difference {worst:.2e}.", flush=True)
    if len(merged) != len(frozen.loc[frozen.cohort.isin(SITES)]) or worst > 1e-8:
        raise ValueError("The rescored GBSA probabilities do not reproduce the frozen calibration file.")


def logistic_update(z, y, w, slope):
    use = w > 0
    kwargs = dict(family=sm.families.Binomial(), freq_weights=w[use])
    if slope:
        fit = sm.GLM(y[use], sm.add_constant(z[use]), **kwargs).fit(maxiter=100)
        return float(fit.params[0]), float(fit.params[1])
    fit = sm.GLM(y[use], np.ones((int(use.sum()), 1)), offset=z[use], **kwargs).fit(maxiter=100)
    return float(fit.params[0]), 1.0


def _fit_frame(frame, horizon, slope):
    p = np.clip(frame[f"risk_{horizon}"].to_numpy(), 1e-6, 1 - 1e-6)
    y, w, _ = ipcw(frame.PFS_TIME_DAYS.to_numpy(), frame.PFS_EVENT.to_numpy(), horizon, np.ones(len(frame), int))
    return logistic_update(logit(p), y, w, slope)


def recalibrate(d, horizon, slope, folds=0, transfer_frame=None, seed=42):
    p = np.clip(d[f"risk_{horizon}"].to_numpy(), 1e-6, 1 - 1e-6)
    z = logit(p)
    if folds:
        codes, ids = pd.factorize(d.PATIENT_ID, sort=True)
        assignment = np.random.default_rng(seed).permutation(len(ids)) % folds
        fold_of_row = assignment[codes]
        out, params = np.empty(len(d)), []
        for k in range(folds):
            train, test = fold_of_row != k, fold_of_row == k
            a, b = _fit_frame(d.loc[train], horizon, slope)
            out[test] = expit(a + b * z[test])
            params.append((a, b))
        return out, params
    a, b = _fit_frame(d if transfer_frame is None else transfer_frame, horizon, slope)
    return expit(a + b * z), [(a, b)]


def recalibration_task(site, horizon, variant):
    frozen = pd.read_csv(FROZEN)
    d = frozen.loc[frozen.cohort.eq(site)].reset_index(drop=True)
    other = frozen.loc[frozen.cohort.eq([s for s in SITES if s != site][0])].reset_index(drop=True)
    spec = dict(VARIANTS[variant])
    transfer = other if spec.pop("transfer", False) else None
    risk, params = recalibrate(d, horizon, spec["slope"], spec.get("folds", 0), transfer)
    d[f"risk_{horizon}"] = risk
    row, _, boot, failures = analyze_horizon(d, site, horizon, N_BOOT, 20261009 + horizon)
    row.update(variant=variant, n_failed_resamples=len(failures),
               update_intercept=float(np.mean([a for a, _ in params])),
               update_slope=float(np.mean([b for _, b in params])),
               n_updates=len(params))
    boot.insert(0, "variant", variant)
    return row, boot


def recalibration(force=False):
    path = OUT / "recalibration_metrics.csv"
    if path.exists() and not force:
        print("Reuse saved recalibration results.", flush=True)
        return
    tasks = [(site, h, v) for site in SITES for h in HORIZONS for v in VARIANTS if VARIANTS[v] is not None]
    rows, boots = [], []
    started = monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for (site, h, v), (row, boot) in zip(tasks, pool.map(recalibration_task, *zip(*tasks))):
            rows.append(row)
            boots.append(boot)
            print(f"{site} {h} d, {v}: ICI {100 * row['ICI']:.2f} [{100 * row['ICI_lo']:.2f}, {100 * row['ICI_hi']:.2f}]; "
                  f"{monotonic() - started:.0f} s.", flush=True)
    original = pd.read_csv(CALIBRATION / "metrics.csv")
    original = original.loc[original.cohort.isin(SITES)].copy()
    original["variant"] = "original"
    metrics = pd.concat([original, pd.DataFrame(rows)], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, index=False)
    original_boot = pd.read_csv(CALIBRATION / "bootstrap_metrics.csv")
    original_boot = original_boot.loc[original_boot.cohort.isin(SITES)].copy()
    original_boot.insert(0, "variant", "original")
    pd.concat([original_boot] + boots, ignore_index=True).to_csv(OUT / "recalibration_bootstrap.csv", index=False)
    recalibration_tertiles()


def recalibration_tertiles():
    frozen = pd.read_csv(FROZEN)
    rows, predictions = [], []
    for site in SITES:
        d = frozen.loc[frozen.cohort.eq(site)].reset_index(drop=True)
        updated = d.copy()
        for h in HORIZONS:
            updated[f"risk_{h}"], _ = recalibrate(d, h, slope=True, folds=CV_FOLDS)
        predictions.append(updated.assign(**{f"original_risk_{h}": d[f"risk_{h}"] for h in HORIZONS}))
        tertile = pd.qcut(d.risk_365, 3, labels=["Low", "Middle", "High"])
        for name, group_idx in d.groupby(tertile, observed=True).groups.items():
            g, u = d.loc[group_idx], updated.loc[group_idx]
            km = KaplanMeierFitter().fit(g.PFS_TIME_DAYS, g.PFS_EVENT)
            for h in HORIZONS:
                rows.append(dict(cohort=site, risk_tertile=name, horizon=h, n_lines=len(g),
                                 n_patients=g.PATIENT_ID.nunique(),
                                 predicted_survival=float((1 - g[f"risk_{h}"]).mean()),
                                 recalibrated_survival=float((1 - u[f"risk_{h}"]).mean()),
                                 observed_KM_survival=float(km.predict(h)),
                                 n_at_risk=int(g.PFS_TIME_DAYS.ge(h).sum())))
    pd.DataFrame(rows).to_csv(OUT / "recalibration_tertiles.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "recalibration_predictions_cv.csv", index=False)


def institution_counts():
    by_fold = pd.read_csv(OUT / "discrimination_by_fold.csv")
    return by_fold.loc[by_fold.model.eq("full_gbsa")].groupby(["endpoint", "site"])[["n_lines", "n_patients", "events"]].first()


def institution_metrics():
    summary = pd.read_csv(OUT / "discrimination_summary.csv")
    full = summary.loc[summary.model.eq("full_gbsa")].set_index(["endpoint", "site", "metric"])
    counts = institution_counts()
    rows = []
    for endpoint in ("pfs_i", "mixed"):
        for site in SITES:
            n = counts.loc[(endpoint, site)]
            rows.append({"endpoint": endpoint, "Institution": site, "Regimens": f"{n.n_lines:,}",
                         "Patients": f"{n.n_patients:,}", "Events": f"{n.events:,}",
                         **{name: f"{full.loc[(endpoint, site, metric), 'point']:.3f} ± "
                                  f"{full.loc[(endpoint, site, metric), 'fold_sd']:.3f}"
                            for metric, name in ED17_METRICS.items()}})
    return pd.DataFrame(rows)


def _ci(s, metric):
    return f"{s.loc[metric, 'point']:.4f} [{s.loc[metric, 'CI_low']:.4f}, {s.loc[metric, 'CI_high']:.4f}]"


def _delta(s, metric):
    return f"{s.loc[metric, 'point']:+.4f} ({s.loc[metric, 'CI_low']:.4f} to {s.loc[metric, 'CI_high']:.4f})"


def convention_section(summary):
    cens = pd.read_csv(OUT / "censoring_summary.csv")
    full = summary.loc[summary.model.eq("full_gbsa")].set_index(["endpoint", "site", "metric"])
    ibs_shift = {site: full.loc[("mixed", site, "IBS"), "point"] - full.loc[("mixed", site, "IBS_train_ipcw"), "point"]
                 for site in SITES}
    parts = [
        "External evaluation convention (Extended Data Table 17 and Results 2.4)",
        "Adopted for every external metric. Each fold model scores the BPC inputs after the preprocessing fitted on "
        "its own MSK-CHORD training rows (Methods 4.4: median and interquartile range from the training fold). The "
        "inverse-probability-of-censoring weights in IBS and cumulative/dynamic AUC come from a Kaplan-Meier estimate "
        "of censoring among the evaluated site's own regimens. The external calibration and recalibration (Results "
        "2.4, Extended Data Table 25) already use this rule: the five-fold mean risks computed here reproduce the "
        "frozen calibration probabilities to within 1e-8. The QC sensitivity (Extended Data Table 15) also estimates "
        "censoring within the evaluated cohort. Internally the weights come from the MSK-CHORD training folds, which "
        "share the censoring process of the lines being scored, so the same rule covers Table 2 and Extended Data "
        "Tables 4 and 13, and no internal value changes.",
        "Previous procedure. The first GENIE scoring of the GBSA fold models fitted one scaler on all "
        "MSK-CHORD lines, because the fold scalers had not been saved, and passed the MSK-CHORD outcomes to the metric "
        "code as the censoring reference, so IBS and AUC were weighted by MSK-CHORD censoring. With fold-k "
        f"preprocessing but MSK-CHORD weights the values are {full.loc[('mixed', 'DFCI', 'IBS_train_ipcw'), 'point']:.4f} "
        f"and {full.loc[('mixed', 'VICC', 'IBS_train_ipcw'), 'point']:.4f} (last column of the discrimination tables "
        "below), so the change in IBS comes from the censoring weights, not from the preprocessing.",
        "Why the weights matter. At time t the IPCW Brier score weights a regimen with an event before t by "
        "1/G(event time) and a regimen still event-free at t by 1/G(t), where G is the probability of remaining "
        "uncensored. A regimen censored before t contributes nothing, and the weights carry its share. The estimate "
        "is consistent only if G describes censoring in the cohort being scored. BPC censors a regimen at the next "
        "regimen start or at last contact, so censoring is about twice as common as in MSK-CHORD and comes earlier "
        "(table below). Under the MSK-CHORD G the weights stay close to 1, the share of the censored regimens is "
        "carried by no one, and the score shrinks toward zero. The shrinkage is larger at VICC, where censoring is "
        f"heaviest (IBS {ibs_shift['VICC']:+.3f} at VICC versus {ibs_shift['DFCI']:+.3f} at DFCI). The previous "
        "values understated absolute error; the difference lies in the weighting, not in the model.",
    ]
    rows = [[r.cohort, r.endpoint, f"{r.lines:,}", f"{r.censored_pct:.1f}", f"{r.uncensored_180:.3f}",
             f"{r.uncensored_365:.3f}"] for r in cens.itertuples()]
    parts += ["Censoring by cohort (Kaplan-Meier probability of remaining uncensored)",
              tabulate(rows, headers=["Cohort", "Endpoint", "Lines", "Censored, %", "Uncensored at 180 d",
                                      "Uncensored at 365 d"], tablefmt="psql", disable_numparse=True)]
    ed17 = institution_metrics()
    parts += ["Extended Data Table 17 values under the adopted convention (full GBSA, mean ± SD across the five fold "
              "models, SD with ddof 0)",
              tabulate(ed17.assign(endpoint=ed17.endpoint.map(ENDPOINTS)), showindex=False,
                       headers=["Endpoint", "Site", "Regimens", "Patients", "Events", "C", "AUC", "AUC at 1 y",
                                "AUC at 2 y", "IBS"], tablefmt="psql", disable_numparse=True)]
    return parts


def generate(context=None):
    summary = pd.read_csv(OUT / "discrimination_summary.csv")
    metrics = pd.read_csv(OUT / "recalibration_metrics.csv")
    avail = pd.read_csv(OUT / "six_variable_availability.csv")
    tert = pd.read_csv(OUT / "recalibration_tertiles.csv")
    counts = institution_counts().loc["mixed"]
    parts = ["External six-variable reference, patient-bootstrap intervals, and site recalibration.",
             "Endpoint: BPC PFS-I + Mixed (28-day minimum, 730-day cap) unless stated; " +
             ", ".join(f"{site} {counts.loc[site, 'n_lines']:,} regimens from {counts.loc[site, 'n_patients']:,} patients"
                       for site in SITES) + ".",
             "Both models are the five MSK-CHORD outer-fold models (the saved GBSA fold models; the six-variable "
             "Cox reference refit on each fold's training rows). Fold-k preprocessing is restored for both. "
             "Point estimates are the mean across the five fold models. Intervals are 1,000 patient-level "
             "bootstrap resamples (seed 42) of that five-fold mean; the same draws are used for both models, so "
             "differences are paired. Predictions stay fixed; intervals do not include model-training uncertainty."]
    parts += convention_section(summary)
    rows = []
    for _, r in avail.iterrows():
        rows.append([r.cohort, f"{r.lines:,}", f"{r.patients:,}", f"{r.first_line_pct:.1f}", f"{r.HR_pos_pct:.1f}",
                     f"{r.HER2_pos_pct:.1f}", f"{r.ECOG_recorded_pct:.1f}", f"{r.liver_in_window_pct:.1f}",
                     f"{r.CA15_3_available_pct:.1f}"])
    parts += ["Availability of the six reference variables (percent of lines)",
              tabulate(rows, headers=["Cohort", "Lines", "Patients", "First line", "HR+", "HER2+", "ECOG recorded",
                                      "Liver, prior 90 d", "CA15-3 available"], tablefmt="psql", disable_numparse=True),
              "ECOG is not released in GENIE BPC, so the external reference sees ECOG 0 for every regimen. "
              "Missing CA15-3 takes the MSK training median after scaling, as in the development cohort."]
    for endpoint, label in ENDPOINTS.items():
        rows = []
        for site in SITES:
            for model, name in MODELS.items():
                s = summary.loc[summary.site.eq(site) & summary.endpoint.eq(endpoint) & summary.model.eq(model)].set_index("metric")
                rows.append([site, name, _ci(s, "C"), f"{s.loc['C', 'fold_sd']:.4f}", _ci(s, "IBS"), _ci(s, "mean_auc"),
                             f"{s.loc['IBS_train_ipcw', 'point']:.4f} / {s.loc['mean_auc_train_ipcw', 'point']:.4f}"])
        parts += [f"External discrimination, {label}",
                  tabulate(rows, headers=["Site", "Model", "C, five-fold mean [95% patient-bootstrap CI]", "Fold SD of C",
                                          "IBS [95% CI]", "Mean AUC [95% CI]", "IBS / mean AUC, MSK-CHORD censoring (previous)"],
                           tablefmt="psql", disable_numparse=True)]
        rows = []
        for site in SITES:
            s = summary.loc[summary.site.eq(site) & summary.endpoint.eq(endpoint) &
                            summary.model.eq("full_gbsa minus reference")].set_index("metric")
            rows.append([site, _delta(s, "C"), f"{s.loc['C', 'fold_min']:+.4f} to {s.loc['C', 'fold_max']:+.4f}",
                         f"{int(s.loc['C', 'folds_favoring_full'])} of 5",
                         f"{100 * s.loc['C', 'bootstrap_share_favoring_full']:.1f}%", _delta(s, "IBS"), _delta(s, "mean_auc")])
        parts += [f"Paired differences, full GBSA minus six-variable reference, {label}",
                  tabulate(rows, headers=["Site", "Change in C (paired 95% CI)", "Fold change range", "Folds favoring full",
                                          "Draws favoring full", "Change in IBS (paired 95% CI)", "Change in mean AUC (paired 95% CI)"],
                           tablefmt="psql", disable_numparse=True)]
    parts += ["Site recalibration of the frozen five-model mean probabilities (log-odds logistic update, IPCW weights).",
              "Apparent: fitted and evaluated on the same site (optimistic). 5-fold CV: patient-level folds within the "
              "site, seed 42, each regimen scored by the update fitted on the other four folds. Other site: the "
              "intercept-and-slope update fitted on all regimens of the other site. Intervals: 1,000 patient resamples "
              "with the recalibrated probabilities held fixed (the CV fit is not repeated inside the bootstrap)."]
    order = list(VARIANTS)
    for measure, label in [("ICI", "ICI (percentage points)"), ("E90", "E90 (percentage points)"),
                           ("citl", "Calibration-in-the-large (log-odds)"), ("slope", "Calibration slope")]:
        rows = []
        for site in SITES:
            for h in HORIZONS:
                m = metrics.loc[metrics.cohort.eq(site) & metrics.horizon.eq(h)].set_index("variant")
                rows.append([site, h] + [cell(m.loc[v], measure) if v in m.index else "" for v in order])
        parts += [label, tabulate(rows, headers=["Site", "Days"] + order, tablefmt="psql", disable_numparse=True)]
    rows = []
    for site in SITES:
        for h in HORIZONS:
            m = metrics.loc[metrics.cohort.eq(site) & metrics.horizon.eq(h)].set_index("variant")
            rows.append([site, h, f"{m.loc['intercept, apparent', 'update_intercept']:+.3f}",
                         f"{m.loc['intercept and slope, apparent', 'update_intercept']:+.3f}",
                         f"{m.loc['intercept and slope, apparent', 'update_slope']:.3f}",
                         f"{m.loc['intercept and slope, 5-fold CV', 'update_intercept']:+.3f}",
                         f"{m.loc['intercept and slope, 5-fold CV', 'update_slope']:.3f}"])
    parts += ["Fitted updates (recalibrated log-odds = intercept + slope x original log-odds)",
              tabulate(rows, headers=["Site", "Days", "Intercept only", "Intercept (with slope)", "Slope",
                                      "CV mean intercept", "CV mean slope"], tablefmt="psql", disable_numparse=True)]
    rows = []
    for site in SITES:
        for name in ("Low", "Middle", "High"):
            g = tert.loc[tert.cohort.eq(site) & tert.risk_tertile.eq(name)].set_index("horizon")
            rows.append([site, name, f"{int(g.n_lines.iloc[0])} ({int(g.n_patients.iloc[0])})"] +
                        [f"{100 * g.loc[h, 'predicted_survival']:.0f} / {100 * g.loc[h, 'recalibrated_survival']:.0f} / "
                         f"{100 * g.loc[h, 'observed_KM_survival']:.0f}" for h in HORIZONS])
    parts += ["Survival by original predicted-risk tertile: mean predicted / mean recalibrated (CV intercept and slope) / observed KM, percent",
              tabulate(rows, headers=["Site", "Tertile", "Regimens (patients)", "180 days", "365 days", "730 days"],
                       tablefmt="psql", disable_numparse=True),
              "Recalculate: python -m analyses.external_validation",
              "Inputs: data/design_matrix.csv, data/genie/genie_design_matrix.csv, results/gbsa/outer*/gbsa-simple-400/model.joblib, "
              f"{FROZEN}, {CALIBRATION / 'metrics.csv'}."]
    return "\n\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute saved results.")
    parser.add_argument("--skip-discrimination", action="store_true")
    parser.add_argument("--skip-recalibration", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    msk = load_msk()
    cohorts = external_cohorts()
    availability(msk, cohorts).to_csv(OUT / "six_variable_availability.csv", index=False)
    censoring(msk, cohorts).to_csv(OUT / "censoring_summary.csv", index=False)
    if not args.skip_discrimination:
        discrimination(msk, cohorts, args.force)
        check_frozen_match()
    if not args.skip_recalibration:
        recalibration(args.force)
    text = generate()
    (OUT / "numbers.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
