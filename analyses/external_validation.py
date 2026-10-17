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

from preprocess import preprocess_df
from analyses.common.auc_bootstrap import bootstrap_auc
from analyses.common.ibs_bootstrap import bootstrap_ibs
from analyses.calibration import FROZEN, HORIZONS, analyze_horizon, ipcw
from analyses.calibration import OUT as CALIBRATION

OUT = Path("results/rebuttal/external")
SITES = ("DFCI", "VICC")
ED17_METRICS = {"C": "C", "mean_auc": "AUC", "auc_at_1y": "AUC@1y", "auc_at_2y": "AUC@2y", "IBS": "IBS"}
N_BOOT = 1000
CV_FOLDS = 5
UPDATED = "intercept and slope, 5-fold CV"


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
    return _trim(gbsa.predict_survival_function(prep["X_test_np"], return_array=True),
                 np.asarray(gbsa.unique_times_, float))


def _trim(curves, grid):
    keep = grid <= 730.0
    return curves[:, keep], grid[keep]


def point_metrics(curves, grid, dm):
    t, e = dm.PFS_TIME_DAYS.to_numpy(float), dm.PFS_EVENT.to_numpy(int)
    c = float(EvalSurv(pd.DataFrame(curves.T, index=grid), t, e).concordance_td("adj_antolini"))
    no_resamples = np.empty((0, dm.PATIENT_ID.nunique()))
    ibs, _ = bootstrap_ibs(curves, grid, dm, no_resamples)
    auc, _ = bootstrap_auc(curves, grid, dm, no_resamples)
    return dict(C=c, IBS=ibs, **auc)


def discrimination(msk, cohorts, force=False):
    path = OUT / "discrimination_by_fold.csv"
    if path.exists() and not force:
        print("Reuse saved discrimination results.", flush=True)
        return
    folds = fold_info()
    rows = []
    for (site, endpoint), cohort in cohorts.items():
        dm = cohort[["PATIENT_ID", "PFS_TIME_DAYS", "PFS_EVENT"]].reset_index(drop=True)
        started = monotonic()
        for fold, train, model_path in folds:
            surv, grid = score_fold(msk, cohort, train, model_path)
            point = point_metrics(surv, grid, dm)
            rows.append(dict(site=site, endpoint=endpoint, fold=fold, model="full_gbsa",
                             n_lines=len(dm), n_patients=dm.PATIENT_ID.nunique(),
                             events=int(dm.PFS_EVENT.sum()), **point))
            print(f"{site} {endpoint} {fold}: C {point['C']:.4f}, IBS {point['IBS']:.4f}, "
                  f"mean AUC {point['mean_auc']:.4f}; {monotonic() - started:.0f} s.", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def logistic_update(z, y, w):
    use = w > 0
    kwargs = dict(family=sm.families.Binomial(), freq_weights=w[use])
    fit = sm.GLM(y[use], sm.add_constant(z[use]), **kwargs).fit(maxiter=100)
    return float(fit.params[0]), float(fit.params[1])


def _fit_frame(frame, horizon):
    p = np.clip(frame[f"risk_{horizon}"].to_numpy(), 1e-6, 1 - 1e-6)
    y, w, _ = ipcw(frame.PFS_TIME_DAYS.to_numpy(), frame.PFS_EVENT.to_numpy(), horizon, np.ones(len(frame), int))
    return logistic_update(logit(p), y, w)


def recalibrate(d, horizon, seed=42):
    p = np.clip(d[f"risk_{horizon}"].to_numpy(), 1e-6, 1 - 1e-6)
    z = logit(p)
    codes, ids = pd.factorize(d.PATIENT_ID, sort=True)
    assignment = np.random.default_rng(seed).permutation(len(ids)) % CV_FOLDS
    fold_of_row = assignment[codes]
    out = np.empty(len(d))
    for k in range(CV_FOLDS):
        train, test = fold_of_row != k, fold_of_row == k
        a, b = _fit_frame(d.loc[train], horizon)
        out[test] = expit(a + b * z[test])
    return out


def recalibration_task(site, horizon):
    frozen = pd.read_csv(FROZEN)
    d = frozen.loc[frozen.cohort.eq(site)].reset_index(drop=True)
    d[f"risk_{horizon}"] = recalibrate(d, horizon)
    row, _, _, failures = analyze_horizon(d, site, horizon, N_BOOT, 20261009 + horizon)
    row.update(variant=UPDATED, n_failed_resamples=len(failures))
    return row


def recalibration(force=False):
    path = OUT / "recalibration_metrics.csv"
    if path.exists() and not force:
        print("Reuse saved recalibration results.", flush=True)
        return
    tasks = [(site, h) for site in SITES for h in HORIZONS]
    rows = []
    started = monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for (site, h), row in zip(tasks, pool.map(recalibration_task, *zip(*tasks))):
            rows.append(row)
            print(f"{site} {h} d, {UPDATED}: ICI {100 * row['ICI']:.2f} [{100 * row['ICI_lo']:.2f}, {100 * row['ICI_hi']:.2f}]; "
                  f"{monotonic() - started:.0f} s.", flush=True)
    original = pd.read_csv(CALIBRATION / "metrics.csv")
    original = original.loc[original.cohort.isin(SITES)].copy()
    original["variant"] = "original"
    metrics = pd.concat([original, pd.DataFrame(rows)], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, index=False)
    recalibration_tertiles()


def recalibration_tertiles():
    frozen = pd.read_csv(FROZEN)
    rows, predictions = [], []
    for site in SITES:
        d = frozen.loc[frozen.cohort.eq(site)].reset_index(drop=True)
        updated = d.copy()
        for h in HORIZONS:
            updated[f"risk_{h}"] = recalibrate(d, h)
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


def institution_metrics():
    by_fold = pd.read_csv(OUT / "discrimination_by_fold.csv")
    full = by_fold.loc[by_fold.model.eq("full_gbsa")]
    rows = []
    for endpoint in ("pfs_i", "mixed"):
        for site in SITES:
            group = full.loc[full.endpoint.eq(endpoint) & full.site.eq(site)]
            n = group.iloc[0]
            rows.append({"endpoint": endpoint, "Institution": site, "Regimens": f"{n.n_lines:,}",
                         "Patients": f"{n.n_patients:,}", "Events": f"{n.events:,}",
                         **{name: f"{group[metric].mean():.3f} ± {group[metric].std(ddof=0):.3f}"
                            for metric, name in ED17_METRICS.items()}})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute saved results.")
    parser.add_argument("--skip-discrimination", action="store_true")
    parser.add_argument("--skip-recalibration", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    msk = load_msk()
    cohorts = external_cohorts()
    pd.DataFrame([dict(cohort=site,
                       CA15_3_available_pct=100 * cohorts[(site, "mixed")].CA15_3_LAST_OBS_LOG.notna().mean())
                  for site in SITES]).to_csv(OUT / "six_variable_availability.csv", index=False)
    if not args.skip_discrimination:
        discrimination(msk, cohorts, args.force)
    if not args.skip_recalibration:
        recalibration(args.force)


if __name__ == "__main__":
    main()
