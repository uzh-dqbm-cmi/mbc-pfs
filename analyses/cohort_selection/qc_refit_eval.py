import argparse
import os
import warnings
from pathlib import Path
from time import monotonic

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import KaplanMeierFitter
from pycox.evaluation import EvalSurv
from scipy.special import logit

from manuscript_utils import load_oof_surv_only
from preprocess import preprocess_df
from analyses.common.ibs_bootstrap import bootstrap_ibs
from analyses.common.survival import ibs_evaluator, load_curves, patient_bootstrap, patient_draws
from analyses.calibration import diagnostic_fits, ipcw, risk_basis
from analyses.cohort_selection.qc_refit import OUT as REFIT, VARIANTS, build_cohort, fold_rows, load_inputs

OUT = REFIT / "eval"
CLASSIFICATION = Path("results/rebuttal/eligibility/late_start/line_classification.csv")
LABELS = {"original": "Original (exclusion kept)", "baseline": "Baseline refit (check)", "late525": "Late-start restored, full PFS (primary)",
          "late525_censor": "Late-start restored, censored at class addition (secondary)",
          "restated_rule": "Restated rule (late-start + GnRH groups restored)", "chemo_endo_all": "Chemo+endocrine rule removed",
          "qc_all": "All QC exclusions restored"}
ICI_REQUESTED = {"HR+/HER2+ first lines starting on chemotherapy"}


def ci(point, samples, digits=4):
    lo, hi = np.quantile(samples, [.025, .975])
    return f"{point:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def surv_at(curves, grid, h):
    j = np.searchsorted(grid, h, side="right") - 1
    return curves[:, max(j, 0)]


def km_at(t, e, h):
    k = KaplanMeierFitter().fit(t, e)
    c = k.confidence_interval_
    row = c.loc[c.index <= h].iloc[-1]
    return float(k.predict(h)), float(row.iloc[0]), float(row.iloc[1])


class Models:
    def __init__(self):
        self.original = pd.read_csv("data/design_matrix.csv")
        self.classes = pd.read_csv(CLASSIFICATION)
        self.sf0, self.grid0 = load_curves(self.original, "full_gbsa")
        with np.load(REFIT / "fixed_original" / "curves.npz") as f:
            fixed = f["surv"]
        meta = pd.read_csv(REFIT / "fixed_original" / "added_meta.csv")
        self.fixed = pd.DataFrame(fixed, index=pd.MultiIndex.from_frame(meta[["PATIENT_ID", "LINE"]]))
        self.cache = {}

    def info(self, name):
        info = pd.read_csv(REFIT / name / "cohort.csv")
        extra = self.classes[["PATIENT_ID", "LINE", "START_CLASS", "SUBTYPE"]].rename(
            columns={"START_CLASS": "START_CLASS_ALL", "SUBTYPE": "SUBTYPE_ALL"})
        info = info.merge(extra, on=["PATIENT_ID", "LINE"], how="left", validate="one_to_one")
        info["FIRST"] = info.LINE.eq(1)
        return info

    def oof(self, name):
        if name in self.cache:
            return self.cache[name]
        info = self.info(name)
        paths = [REFIT / name / f"outer{k}" / "surv_test.npz" for k in range(5)]
        sf, grid = load_oof_surv_only(info, paths)
        assert not np.isnan(sf.to_numpy()).any()
        self.cache[name] = (info, sf.to_numpy(), grid)
        return self.cache[name]

    def original_for(self, info):
        n0 = int(info.SOURCE.eq("retained").sum())
        assert (info.PATIENT_ID.iloc[:n0].to_numpy() == self.original.PATIENT_ID.to_numpy()).all()
        restored = info.iloc[n0:]
        rows = self.fixed.loc[list(zip(restored.PATIENT_ID, restored.LINE))].to_numpy() if len(restored) else np.zeros((0, len(self.grid0)))
        return np.vstack([self.sf0.to_numpy(), rows]), self.grid0


def discrimination(models, n_boot):
    original = models.original
    weights = patient_draws(original, n_boot=n_boot)[2]
    base = patient_bootstrap(models.sf0.to_numpy(), models.grid0, original, weights=weights)
    den = base["den"]
    base_ibs, base_ibs_samples = bootstrap_ibs(models.sf0.to_numpy(), models.grid0, original, weights)
    rows = [{"Model": LABELS["original"], "C on 8,791 lines": f"{base['C']:.4f}",
             "Delta C vs original (paired 95% CI)": "reference",
             "Delta IBS vs original (paired 95% CI)": "reference", "Full-cohort lines": f"{len(original):,}"}]
    for name in (variant for variant in VARIANTS if variant != "baseline"):
        info, curves, grid = models.oof(name)
        n0 = int(info.SOURCE.eq("retained").sum())
        sub = patient_bootstrap(curves[:n0], grid, original, weights=weights, den=den)
        ibs, ibs_samples = bootstrap_ibs(curves[:n0], grid, original, weights)
        d_c, d_ibs = sub["samples"] - base["samples"], ibs_samples - base_ibs_samples
        rows.append({"Model": LABELS[name], "C on 8,791 lines": f"{sub['C']:.4f}",
                     "Delta C vs original (paired 95% CI)": ci(sub["C"] - base["C"], d_c),
                     "Delta IBS vs original (paired 95% CI)": ci(ibs - base_ibs, d_ibs),
                     "Full-cohort lines": f"{len(info):,}"})
        print(f"Discrimination: {name} done", flush=True)
    return pd.DataFrame(rows)


def calibration_point(t, e, p, h):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = logit(p)
    counts = np.ones(len(t), dtype=int)
    try:
        y, w, _ = ipcw(t, e, h, counts)
        basis = np.asarray(risk_basis(p)[2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics, _ = diagnostic_fits(y, w, z, basis, p, counts)
        return dict(citl=metrics[0], slope=metrics[2], ici=metrics[3])
    except Exception:
        try:
            y, w, _ = ipcw(t, e, h, counts)
            use = w > 0
            kwargs = dict(family=sm.families.Binomial(), freq_weights=w[use])
            citl = sm.GLM(y[use], np.ones((use.sum(), 1)), offset=z[use], **kwargs).fit(maxiter=100)
            weak = sm.GLM(y[use], sm.add_constant(z[use]), **kwargs).fit(maxiter=100)
            return dict(citl=float(citl.params[0]), slope=float(weak.params[1]), ici=np.nan)
        except Exception:
            return dict(citl=np.nan, slope=np.nan, ici=np.nan)


def strata(info):
    s = info.SUBTYPE_ALL
    chemo = info.START_CLASS_ALL.eq("Chemotherapy")
    endo = info.START_CLASS_ALL.eq("Endocrine")
    restored = info.SOURCE.eq("restored")
    return [("Restored lines, all", restored),
            ("HR+/HER2+ first lines starting on chemotherapy", s.eq("HR+/HER2+") & info.FIRST & chemo),
            ("HR+/HER2- first lines starting on chemotherapy", s.eq("HR+/HER2-") & info.FIRST & chemo),
            ("HR+/HER2+ first lines, all", s.eq("HR+/HER2+") & info.FIRST),
            ("HR+ endocrine-start lines (incl. restored endocrine-start lines)", s.isin(["HR+/HER2-", "HR+/HER2+"]) & endo),
            ("Unaffected: TNBC", s.eq("TNBC")), ("Unaffected: HR-/HER2+", s.eq("HR-/HER2+")),
            ("All lines in cohort", pd.Series(True, index=info.index))]


def calibration(models):
    rows = []
    for name in ("late525",):
        info, curves, grid = models.oof(name)
        orig, grid0 = models.original_for(info)
        t, e = info.PFS_TIME_DAYS.to_numpy(float), info.PFS_EVENT.to_numpy(int)
        for label, mask in strata(info):
            mask = mask.to_numpy()
            n = int(mask.sum())
            if n == 0:
                continue
            for model_label, c, g in [(LABELS["original"], orig, grid0), (LABELS[name], curves, grid)]:
                row = {"Cohort": LABELS[name], "Stratum": label, "Model": model_label, "Lines": f"{n:,}", "Patients": f"{info.PATIENT_ID[mask].nunique():,}"}
                degenerate = e[mask].sum() == 0
                for h in (365,):
                    s_pred = surv_at(c[mask], g, h)
                    if degenerate:
                        row[f"{h} d: KM S [95% CI]"] = "no events (censored at class addition)"
                        row[f"{h} d: mean predicted S"] = f"{s_pred.mean():.3f}"
                        for k in ("predicted - KM (points)", "at risk", "CITL", "slope", "ICI (points)"):
                            row[f"{h} d: {k}"] = "-"
                        continue
                    km, lo, hi = km_at(t[mask], e[mask], h)
                    at_risk = int((t[mask] >= h).sum())
                    cal = calibration_point(t[mask], e[mask], 1 - s_pred, h) if n >= 100 else dict(citl=np.nan, slope=np.nan, ici=np.nan)
                    row[f"{h} d: KM S [95% CI]"] = f"{km:.3f} [{lo:.3f}, {hi:.3f}]"
                    row[f"{h} d: mean predicted S"] = f"{s_pred.mean():.3f}"
                    row[f"{h} d: predicted - KM (points)"] = f"{100 * (s_pred.mean() - km):+.1f}"
                    row[f"{h} d: at risk"] = f"{at_risk:,}"
                    row[f"{h} d: CITL"] = f"{cal['citl']:+.3f}" if np.isfinite(cal["citl"]) else "-"
                    row[f"{h} d: slope"] = f"{cal['slope']:.3f}" if np.isfinite(cal["slope"]) else "-"
                    row[f"{h} d: ICI (points)"] = f"{100 * cal['ici']:.2f}" if (np.isfinite(cal["ici"]) and (n >= 300 or label in ICI_REQUESTED)) else "-"
                rows.append(row)
        print(f"Calibration: {name} done", flush=True)
    return pd.DataFrame(rows)


def external_cohorts():
    genie = pd.read_csv("data/genie/genie_design_matrix.csv")
    genie["site"] = genie.PATIENT_ID.str.split("-").str[1]
    frozen = pd.read_csv("data/rebuttal/calibration/horizon_predictions.csv")
    out = {}
    for site in ("DFCI", "VICC"):
        sub = genie.loc[genie.site.eq(site)].drop(columns="site")
        labels = frozen.loc[frozen.cohort.eq(site), ["PATIENT_ID", "LINE", "PFS_TIME_DAYS", "PFS_EVENT"]]
        mixed = sub.drop(columns=["PFS_TIME_DAYS", "PFS_EVENT"]).merge(labels, on=["PATIENT_ID", "LINE"], how="inner")
        assert len(mixed) == len(labels), (site, len(mixed), len(labels))
        out[site] = mixed.reset_index(drop=True)
    return out


def external():
    original, added, meta, folds, splits = load_inputs()
    cohorts = external_cohorts()
    daily = np.arange(1, 730, dtype=float)
    rows = []
    for name in ("baseline", "late525"):
        features, info = build_cohort(name, original, added, meta)
        for site, ext in cohorts.items():
            avg = np.zeros((len(ext), len(daily)))
            combined = pd.concat([features, ext.reindex(columns=features.columns)], ignore_index=True)
            idx = np.arange(len(features), len(combined))
            for fold in folds:
                train, _ = fold_rows(splits[fold], info, fold)
                prep = preprocess_df(train, idx, idx, combined)
                model = joblib.load(REFIT / name / fold / "model.joblib")
                s = model.predict_survival_function(prep["X_test_np"], return_array=True)
                g = np.asarray(model.unique_times_, float)
                j = np.searchsorted(g, daily, side="right") - 1
                step = np.where(j[None, :] >= 0, s[:, np.clip(j, 0, None)], 1.0)
                avg += step / len(folds)
            t, e = ext.PFS_TIME_DAYS.to_numpy(float), ext.PFS_EVENT.to_numpy(int)
            c = float(EvalSurv(pd.DataFrame(avg.T, index=daily), t, e).concordance_td("adj_antolini"))
            ibs = float(ibs_evaluator(avg, daily, ext).integrated_brier_score())
            row = {"Model": LABELS["original"] if name == "baseline" else LABELS[name], "Site": site, "Regimens": f"{len(ext):,}",
                   "Patients": f"{ext.PATIENT_ID.nunique():,}", "C": f"{c:.4f}", "IBS": f"{ibs:.4f}"}
            for h in (180,):
                s_pred = surv_at(avg, daily, h)
                km = km_at(t, e, h)[0]
                cal = calibration_point(t, e, 1 - s_pred, h)
                row[f"{h} d: KM S"] = f"{km:.3f}"; row[f"{h} d: mean predicted S"] = f"{s_pred.mean():.3f}"
                row[f"{h} d: CITL"] = f"{cal['citl']:+.3f}"; row[f"{h} d: slope"] = f"{cal['slope']:.3f}"
                row[f"{h} d: ICI (points)"] = f"{100 * cal['ici']:.2f}" if np.isfinite(cal["ici"]) else "-"
            rows.append(row)
            print(f"External evaluation: {name} {site} C={c:.4f} IBS={ibs:.4f}", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    models = Models()
    discrimination(models, args.n_boot).to_csv(OUT / "discrimination.csv", index=False)
    calibration(models).to_csv(OUT / "calibration_by_stratum.csv", index=False)
    external().to_csv(OUT / "external.csv", index=False)
    print(f"Finished in {monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
