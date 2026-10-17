import argparse
import json
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
from scipy.stats import spearmanr
from tabulate import tabulate

from config import CALIBRATION_HORIZONS as HORIZONS
from manuscript_utils import assign_tertiles, load_oof_surv_only
from preprocess import preprocess_df
from analyses.common.ibs_bootstrap import bootstrap_ibs
from analyses.common.survival import ibs_evaluator, load_curves, patient_bootstrap, patient_draws
from analyses.calibration import diagnostic_fits, ipcw, risk_basis
from analyses.cohort_selection.qc_refit import OUT as REFIT, VARIANTS, build_cohort, fold_rows, load_inputs
from survival_metrics import rmst_from_curve

OUT = REFIT / "eval"
CLASSIFICATION = Path("results/rebuttal/eligibility/late_start/line_classification.csv")
LABELS = {"original": "Original (exclusion kept)", "baseline": "Baseline refit (check)", "late525": "Late-start restored, full PFS (primary)",
          "late525_censor": "Late-start restored, censored at class addition (secondary)",
          "restated_rule": "Restated rule (late-start + GnRH groups restored)", "chemo_endo_all": "Chemo+endocrine rule removed",
          "qc_all": "All QC exclusions restored"}
REFIT_VARIANTS = [v for v in VARIANTS if v != "baseline"]
CHEMO_START = "Chemotherapy start, endocrine added"
ICI_REQUESTED = {"HR+/HER2+ first lines starting on chemotherapy"}
ENDO_START = "Endocrine start, chemotherapy added"


def fmt(frame):
    return tabulate(frame, headers="keys", showindex=False, tablefmt="psql", disable_numparse=True)


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
        self.ref0, self.ref_grid0 = load_curves(self.original, "ref_mlot1_ca15_3")
        with np.load(REFIT / "fixed_original" / "curves.npz") as f:
            fixed = f["surv"]
        meta = pd.read_csv(REFIT / "fixed_original" / "added_meta.csv")
        self.fixed = pd.DataFrame(fixed, index=pd.MultiIndex.from_frame(meta[["PATIENT_ID", "LINE"]]))
        self.cache = {}

    def info(self, name):
        info = pd.read_csv(REFIT / name / "cohort.csv")
        extra = self.classes[["PATIENT_ID", "LINE", "START_CLASS", "SUBTYPE", "CATEGORY", "LATE_START"]].rename(
            columns={"START_CLASS": "START_CLASS_ALL", "SUBTYPE": "SUBTYPE_ALL", "CATEGORY": "CATEGORY_ALL", "LATE_START": "LATE_ALL"})
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

    def reference_oof(self, name):
        info = self.info(name)
        paths = [REFIT / name / f"outer{k}" / "surv_test_reference.npz" for k in range(5)]
        sf, grid = load_oof_surv_only(info, paths)
        return sf.to_numpy(), grid

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
    rows = [{"Model": LABELS["original"], "Lines": f"{len(original):,}", "C on 8,791 lines": f"{base['C']:.4f}",
             "Delta C vs original (paired 95% CI)": "reference", "IBS on 8,791 lines": f"{base_ibs:.4f}",
             "Delta IBS vs original (paired 95% CI)": "reference", "Full-cohort lines": f"{len(original):,}",
             "Full-cohort C (95% CI)": f"{base['C']:.4f} [{np.quantile(base['samples'], .025):.4f}, {np.quantile(base['samples'], .975):.4f}]",
             "Full-cohort IBS": f"{base_ibs:.4f}", "C in restored lines": "-"}]
    deltas = {}
    for name in VARIANTS:
        info, curves, grid = models.oof(name)
        n0 = int(info.SOURCE.eq("retained").sum())
        sub = patient_bootstrap(curves[:n0], grid, original, weights=weights, den=den)
        ibs, ibs_samples = bootstrap_ibs(curves[:n0], grid, original, weights)
        d_c, d_ibs = sub["samples"] - base["samples"], ibs_samples - base_ibs_samples
        deltas[name] = dict(delta_C=sub["C"] - base["C"], delta_C_lo=np.quantile(d_c, .025), delta_C_hi=np.quantile(d_c, .975),
                            delta_IBS=ibs - base_ibs, delta_IBS_lo=np.quantile(d_ibs, .025), delta_IBS_hi=np.quantile(d_ibs, .975))
        full = patient_bootstrap(curves, grid, info, n_boot=n_boot)
        full_ibs = float(ibs_evaluator(curves, grid, info).integrated_brier_score())
        restored = info.index[info.SOURCE.eq("restored")]
        c_restored = (EvalSurv(pd.DataFrame(curves[restored].T, index=grid), info.PFS_TIME_DAYS.to_numpy()[restored],
                               info.PFS_EVENT.to_numpy()[restored]).concordance_td("adj_antolini")
                      if len(restored) and info.PFS_EVENT.to_numpy()[restored].sum() > 0 else np.nan)
        rows.append({"Model": LABELS[name], "Lines": f"{len(original):,}", "C on 8,791 lines": f"{sub['C']:.4f}",
                     "Delta C vs original (paired 95% CI)": ci(sub["C"] - base["C"], d_c),
                     "IBS on 8,791 lines": f"{ibs:.4f}", "Delta IBS vs original (paired 95% CI)": ci(ibs - base_ibs, d_ibs),
                     "Full-cohort lines": f"{len(info):,}",
                     "Full-cohort C (95% CI)": f"{full['C']:.4f} [{np.quantile(full['samples'], .025):.4f}, {np.quantile(full['samples'], .975):.4f}]",
                     "Full-cohort IBS": f"{full_ibs:.4f}", "C in restored lines": f"{c_restored:.4f}" if np.isfinite(c_restored) else "-"})
        print(f"part 1: {name} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(deltas).T.rename_axis("variant").reset_index()


def stability(models):
    rows = []
    for name in VARIANTS:
        info, curves, grid = models.oof(name)
        n0 = int(info.SOURCE.eq("retained").sum())
        row = {"Model": LABELS[name]}
        for h in HORIZONS:
            r0 = 1 - surv_at(models.sf0.to_numpy(), models.grid0, h)
            r1 = 1 - surv_at(curves[:n0], grid, h)
            t0, t1 = assign_tertiles(pd.Series(r0)), assign_tertiles(pd.Series(r1))
            row[f"Spearman rho, {h}-d risk"] = f"{spearmanr(r0, r1).statistic:.4f}"
            row[f"Lines changing tertile, {h} d"] = f"{100 * (t0 != t1).mean():.1f}%"
            row[f"Mean |delta risk|, {h} d (pct points)"] = f"{100 * np.abs(r1 - r0).mean():.2f}"
            if h == 365:
                row["Median |delta risk|, 365 d (pct points)"] = f"{100 * np.median(np.abs(r1 - r0)):.2f}"
                row["Lines with |delta| > 5 points, 365 d"] = f"{100 * (np.abs(r1 - r0) > .05).mean():.1f}%"
        rows.append(row)
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
            ("Restored: chemotherapy start, endocrine added", restored & info.CATEGORY_ALL.eq(CHEMO_START)),
            ("Restored: endocrine start, chemotherapy added", restored & info.CATEGORY_ALL.eq(ENDO_START)),
            ("Restored: HR+/HER2+ first lines, chemotherapy start", restored & s.eq("HR+/HER2+") & info.FIRST & info.CATEGORY_ALL.eq(CHEMO_START)),
            ("Restored: HR+/HER2- first lines, chemotherapy start", restored & s.eq("HR+/HER2-") & info.FIRST & info.CATEGORY_ALL.eq(CHEMO_START)),
            ("HR+/HER2+ first lines starting on chemotherapy", s.eq("HR+/HER2+") & info.FIRST & chemo),
            ("HR+/HER2- first lines starting on chemotherapy", s.eq("HR+/HER2-") & info.FIRST & chemo),
            ("HR+/HER2+ first lines, all", s.eq("HR+/HER2+") & info.FIRST),
            ("HR+/HER2+ all lines", s.eq("HR+/HER2+")),
            ("HR+ endocrine-start lines (incl. restored endocrine-start lines)", s.isin(["HR+/HER2-", "HR+/HER2+"]) & endo),
            ("Unaffected: HR+ endocrine-start lines, retained only", s.isin(["HR+/HER2-", "HR+/HER2+"]) & endo & ~restored),
            ("Unaffected: TNBC", s.eq("TNBC")), ("Unaffected: HR-/HER2+", s.eq("HR-/HER2+")),
            ("All lines in cohort", pd.Series(True, index=info.index))]


def calibration(models, variants=("late525", "late525_censor", "restated_rule", "chemo_endo_all")):
    rows = []
    for name in variants:
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
                for h in HORIZONS:
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
        print(f"part 3: {name} done", flush=True)
    return pd.DataFrame(rows)


def restored_side_by_side(models):
    info, _, _ = models.oof("late525")
    restored = info.SOURCE.eq("restored").to_numpy()
    info_c, _, _ = models.oof("late525_censor")
    t, e = info.PFS_TIME_DAYS.to_numpy(float)[restored], info.PFS_EVENT.to_numpy(int)[restored]
    tc, ec = info_c.PFS_TIME_DAYS.to_numpy(float)[restored], info_c.PFS_EVENT.to_numpy(int)[restored]
    rows = []
    for label, cat in [("All restored", None), ("Chemotherapy start", CHEMO_START), ("Endocrine start", ENDO_START)]:
        sel = np.ones(restored.sum(), bool) if cat is None else info.CATEGORY_ALL[restored].eq(cat).to_numpy()
        row = {"Restored lines": label, "n": f"{int(sel.sum()):,}"}
        for h in HORIZONS:
            row[f"{h} d: KM, full PFS"] = f"{km_at(t[sel], e[sel], h)[0]:.3f}"
            row[f"{h} d: KM, censored at class addition"] = f"{km_at(tc[sel], ec[sel], h)[0]:.3f}"
            for name in ["original", "late525", "late525_censor", "restated_rule", "chemo_endo_all"]:
                if name == "original":
                    c, g = models.original_for(info)
                else:
                    _, c, g = models.oof(name)
                    i2 = models.info(name)
                    key = list(zip(info.PATIENT_ID[restored], info.LINE[restored]))
                    pos = pd.Series(np.arange(len(i2)), index=pd.MultiIndex.from_frame(i2[["PATIENT_ID", "LINE"]])).loc[key].to_numpy()
                    c = c[pos]
                    row[f"{h} d: mean S, {name}"] = f"{surv_at(c[sel], g, h).mean():.3f}"
                    continue
                row[f"{h} d: mean S, {name}"] = f"{surv_at(c[restored][sel], g, h).mean():.3f}"
        rows.append(row)
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


def external(models, variants=("baseline", "late525", "late525_censor", "restated_rule", "chemo_endo_all")):
    original, added, meta, folds, splits = load_inputs()
    cohorts = external_cohorts()
    daily = np.arange(1, 730, dtype=float)
    rows = []
    for name in variants:
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
            for h in HORIZONS:
                s_pred = surv_at(avg, daily, h)
                km = km_at(t, e, h)[0]
                cal = calibration_point(t, e, 1 - s_pred, h)
                row[f"{h} d: KM S"] = f"{km:.3f}"; row[f"{h} d: mean predicted S"] = f"{s_pred.mean():.3f}"
                row[f"{h} d: CITL"] = f"{cal['citl']:+.3f}"; row[f"{h} d: slope"] = f"{cal['slope']:.3f}"
                row[f"{h} d: ICI (points)"] = f"{100 * cal['ici']:.2f}" if np.isfinite(cal["ici"]) else "-"
            rows.append(row)
            print(f"part 4: {name} {site} C={c:.4f} IBS={ibs:.4f}", flush=True)
    return pd.DataFrame(rows)


def tertiles(models):
    rows = []
    for name in ["original"] + REFIT_VARIANTS:
        if name == "original":
            info, curves, grid = models.info("baseline"), models.sf0.to_numpy(), models.grid0
        else:
            info, curves, grid = models.oof(name)
        for scope in ["8,791 original lines", "full cohort"]:
            sel = info.SOURCE.eq("retained").to_numpy() if scope.startswith("8,791") else np.ones(len(info), bool)
            if name == "original" and scope == "full cohort":
                continue
            risk = 1 - surv_at(curves[sel], grid, 365)
            tert = assign_tertiles(pd.Series(risk)).to_numpy()
            t, e = info.PFS_TIME_DAYS.to_numpy(float)[sel], info.PFS_EVENT.to_numpy(int)[sel]
            row = {"Model": LABELS[name], "Scope": scope, "Lines": f"{int(sel.sum()):,}"}
            for tt in ["low", "mid", "high"]:
                m = tert == tt
                k = KaplanMeierFitter().fit(t[m], e[m])
                km365 = float(k.predict(365))
                sf = k.survival_function_
                rmst = rmst_from_curve(sf.index.to_numpy(float), sf.iloc[:, 0].to_numpy(float), 365)
                pred = surv_at(curves[sel][m], grid, 365).mean()
                row[f"{tt}: KM S(365)"] = f"{km365:.3f}"; row[f"{tt}: mean pred S(365)"] = f"{pred:.3f}"; row[f"{tt}: KM RMST@365 (d)"] = f"{rmst:.0f}"
            lo, hi = tert == "low", tert == "high"
            row["KM S(365) low - high"] = f"{float(KaplanMeierFitter().fit(t[lo], e[lo]).predict(365)) - float(KaplanMeierFitter().fit(t[hi], e[hi]).predict(365)):.3f}"
            rows.append(row)
    return pd.DataFrame(rows)


def importances(models):
    folds = json.loads(Path("results/gbsa/eval_metrics.json").read_text())["outer_folds"]
    def mean_importance(paths):
        frames = [pd.Series(dict(json.load(open(p)))) for p in paths]
        return pd.concat(frames, axis=1).fillna(0).mean(axis=1)
    orig = mean_importance([Path(info["results_path"]) / "gbsa_built_in_feature_importance.json" for info in folds.values()])
    table = pd.DataFrame({"original": orig})
    for name in REFIT_VARIANTS:
        table[name] = mean_importance([REFIT / name / f"outer{k}" / "importance.json" for k in range(5)])
    table = table.fillna(0)
    ranks = table.rank(ascending=False)
    top = table.original.sort_values(ascending=False).head(15).index
    rows = []
    for f in top:
        row = {"Feature": f, "Original importance": f"{table.original[f]:.4f}", "Original rank": int(ranks.original[f])}
        for name in ["late525", "late525_censor", "restated_rule", "chemo_endo_all"]:
            row[f"{name} rank"] = int(ranks[name][f])
        rows.append(row)
    corr = pd.DataFrame([{"Comparison": f"original vs {name}", "Spearman rho of mean importances": f"{spearmanr(table.original, table[name]).statistic:.3f}",
                          "Top-15 overlap": f"{len(set(top) & set(table[name].sort_values(ascending=False).head(15).index))} of 15"}
                         for name in REFIT_VARIANTS])
    return pd.DataFrame(rows), corr


def reference_gain(models, n_boot):
    original = models.original
    weights = patient_draws(original, n_boot=n_boot)[2]
    ref = patient_bootstrap(models.ref0.to_numpy(), models.ref_grid0, original, weights=weights)
    den = ref["den"]
    rows = []
    for name in ["original"] + REFIT_VARIANTS:
        if name == "original":
            full = patient_bootstrap(models.sf0.to_numpy(), models.grid0, original, weights=weights, den=den)
            d = full["samples"] - ref["samples"]
            rows.append({"Model": LABELS[name], "Scope": "8,791 original lines", "GBSA C": f"{full['C']:.4f}", "Reference C": f"{ref['C']:.4f}",
                         "Gain in C (paired 95% CI)": ci(full["C"] - ref["C"], d)})
            continue
        info, curves, grid = models.oof(name)
        n0 = int(info.SOURCE.eq("retained").sum())
        full = patient_bootstrap(curves[:n0], grid, original, weights=weights, den=den)
        d = full["samples"] - ref["samples"]
        rows.append({"Model": LABELS[name], "Scope": "8,791 original lines (saved reference)", "GBSA C": f"{full['C']:.4f}", "Reference C": f"{ref['C']:.4f}",
                     "Gain in C (paired 95% CI)": ci(full["C"] - ref["C"], d)})
        rc, rgrid = models.reference_oof(name)
        w_v = patient_draws(info, n_boot=n_boot)[2]
        ref_v = patient_bootstrap(rc, rgrid, info, weights=w_v)
        full_v = patient_bootstrap(curves, grid, info, weights=w_v, den=ref_v["den"])
        d = full_v["samples"] - ref_v["samples"]
        rows.append({"Model": LABELS[name], "Scope": f"full cohort ({len(info):,} lines; reference refit)", "GBSA C": f"{full_v['C']:.4f}",
                     "Reference C": f"{ref_v['C']:.4f}", "Gain in C (paired 95% CI)": ci(full_v["C"] - ref_v["C"], d)})
        print(f"part 5: gain {name} done", flush=True)
    return pd.DataFrame(rows)


def surv_at_each(curves, grid, times):
    j = np.clip(np.searchsorted(grid, times, side="right") - 1, 0, None)
    return curves[np.arange(len(curves)), j]


def conditional_restored(models):
    info, curves, grid = models.oof("late525")
    orig, grid0 = models.original_for(info)
    restored = info.SOURCE.eq("restored").to_numpy()
    add = info.ADD_DAY.to_numpy(float)
    t, e = info.PFS_TIME_DAYS.to_numpy(float), info.PFS_EVENT.to_numpy(int)
    s = info.SUBTYPE_ALL
    groups = [("All restored lines", restored),
              ("Chemotherapy start, endocrine therapy added", restored & info.CATEGORY_ALL.eq(CHEMO_START).to_numpy()),
              ("Endocrine start, chemotherapy added", restored & info.CATEGORY_ALL.eq(ENDO_START).to_numpy()),
              ("HR+/HER2+ first lines, chemotherapy start", restored & s.eq("HR+/HER2+").to_numpy() & info.FIRST.to_numpy() & info.CATEGORY_ALL.eq(CHEMO_START).to_numpy()),
              ("HR+/HER2- first lines, chemotherapy start", restored & s.eq("HR+/HER2-").to_numpy() & info.FIRST.to_numpy() & info.CATEGORY_ALL.eq(CHEMO_START).to_numpy())]
    rows = []
    for label, mask in groups:
        for model_label, c, g in [(LABELS["original"], orig, grid0), (LABELS["late525"], curves, grid)]:
            row = {"Restored stratum": label, "Model": model_label, "Lines": f"{int(mask.sum()):,}",
                   "Second-class start, median day": f"{np.median(add[mask]):.0f}"}
            s_add = surv_at_each(c[mask], g, add[mask])
            for h in HORIZONS:
                s_h = surv_at(c[mask], g, h)
                cond = np.where(add[mask] >= h, 1.0, s_h / s_add)
                km = km_at(t[mask], e[mask], h)[0]
                row[f"{h} d: KM S"] = f"{km:.3f}"
                row[f"{h} d: predicted S, unconditional"] = f"{s_h.mean():.3f}"
                row[f"{h} d: predicted S, conditional on reaching the second class"] = f"{cond.mean():.3f}"
                row[f"{h} d: gap unconditional / conditional (points)"] = f"{100 * (s_h.mean() - km):+.1f} / {100 * (cond.mean() - km):+.1f}"
            rows.append(row)
    return pd.DataFrame(rows)


PART_TEXT = {
    "discrimination": "Part 1. Pooled out-of-fold C (adjusted Antolini) and IBS. Paired intervals: 1,000 patient draws (seed 42), each sampled patient "
                      "contributing all lines, predictions fixed; the same draws for every model. Full-cohort C intervals resample that cohort's patients. "
                      "IBS integrates over the evaluated cohort's observed censoring times.",
    "stability": "Part 2. Prediction stability on the 8,791 original lines: original out-of-fold risk versus refit out-of-fold risk "
                 "(risk = 1 - S(h), step evaluation at the last grid time <= h). Tertiles: equal-frequency thirds of each model's own risk.",
    "calibration_by_stratum": "Part 3. Calibration by stratum. KM: Kaplan-Meier survival with 95% CI against the labels of the evaluated cohort "
                              "(for the censored variant, restored lines are censored at the class addition). CITL and slope: IPCW logistic calibration on the "
                              "log-odds scale (ideal 0 and 1; n >= 100). ICI: mean absolute error of the three-knot spline calibration curve in percentage points "
                              "(n >= 300, and HR+/HER2+ first lines starting on chemotherapy, 159 lines, on request). "
                              "Original-model predictions for restored lines come from the saved fold models.",
    "conditional_restored": "Part 3c. Restored late-start lines: predictions conditioned on survival to each line's own second-class start "
                            "(S(h)/S(day of addition) when the addition precedes h, otherwise 1) against the observed Kaplan-Meier survival; "
                            "the remaining gap is the part of the selection effect that is not arithmetic.",
    "restored_side_by_side": "Part 3b. Restored late-start lines: observed survival under both label versions and every model's mean predicted survival.",
    "external": "Part 4. External DFCI and VICC regimens (BPC PFS-I + Mixed, 28-day minimum, 730-day cap; frozen labels). Each variant's "
                "five fold models are applied with fold-specific preprocessing and their survival curves averaged on a daily grid, as for the "
                "original. C: adjusted Antolini. IBS: censoring weights from the site's own regimens. Calibration as in Part 3.",
    "tertiles": "Part 5a. Tertile separation: equal-frequency tertiles of each model's own 365-day risk; KM survival and RMST at 365 days per tertile.",
    "importance_top15": "Part 5b. Built-in GBSA feature importances averaged over the five fold models (original top 15 with ranks under each refit).",
    "importance_correlation": "",
    "reference_gain": "Part 5c. Gain over the six-variable Cox reference (first-line status, HR, HER2, latest ECOG, liver involvement, latest log CA15-3). "
                      "On the 8,791 lines the saved reference predictions are used; on each full cohort the reference is refit on the same folds.",
}
PART_FILES = ["discrimination", "stability", "calibration_by_stratum", "restored_side_by_side", "conditional_restored", "external", "tertiles",
              "importance_top15", "importance_correlation", "reference_gain"]


def assemble():
    parts = []
    for name in PART_FILES:
        path = OUT / f"{name}.csv"
        if not path.exists():
            parts.append(f"[{name}: not computed]")
            continue
        frame = pd.read_csv(path, dtype=str).fillna("")
        parts += [t for t in [PART_TEXT[name]] if t] + [fmt(frame)]
    parts.append("Recalculate: python -m analyses.cohort_selection.qc_refit_eval --parts all")
    text = "\n\n".join(parts) + "\n"
    (OUT / "qc_refit_eval.txt").write_text(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", default="all", help="comma list of 1,2,3,3c,4,5a,5b,5c, or all; 'assemble' only writes the report")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {"1", "2", "3", "3c", "4", "5a", "5b", "5c"} if args.parts == "all" else set(args.parts.split(","))
    started = monotonic()
    if parts - {"assemble"}:
        models = Models()
    if "1" in parts:
        disc, deltas = discrimination(models, args.n_boot)
        disc.to_csv(OUT / "discrimination.csv", index=False); deltas.to_csv(OUT / "paired_deltas.csv", index=False)
    if "2" in parts:
        stability(models).to_csv(OUT / "stability.csv", index=False)
    if "3" in parts:
        calibration(models).to_csv(OUT / "calibration_by_stratum.csv", index=False)
        restored_side_by_side(models).to_csv(OUT / "restored_side_by_side.csv", index=False)
    if "3c" in parts:
        conditional_restored(models).to_csv(OUT / "conditional_restored.csv", index=False)
    if "4" in parts:
        external(models).to_csv(OUT / "external.csv", index=False)
    if "5a" in parts:
        tertiles(models).to_csv(OUT / "tertiles.csv", index=False)
    if "5b" in parts:
        imp, corr = importances(models)
        imp.to_csv(OUT / "importance_top15.csv", index=False); corr.to_csv(OUT / "importance_correlation.csv", index=False)
    if "5c" in parts:
        reference_gain(models, args.n_boot).to_csv(OUT / "reference_gain.csv", index=False)
    print(f"parts {sorted(parts)} finished in {monotonic() - started:.0f} s", flush=True)
    print(assemble())


if __name__ == "__main__":
    main()
