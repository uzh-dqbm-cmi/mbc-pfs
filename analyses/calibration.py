import warnings
from time import monotonic
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import KaplanMeierFitter
from patsy import build_design_matrices, dmatrix
from scipy.special import logit

from config import CALIBRATION_HORIZONS as HORIZONS

FROZEN = Path("data/rebuttal/calibration/horizon_predictions.csv")
OUT = Path("results/rebuttal/calibration")
COHORTS = ("MSK_OOF", "DFCI", "VICC")
METRICS = ("citl", "intercept", "slope", "ICI", "E50", "E90")
SCALE = {"citl": (1, 3), "slope": (1, 3), "ICI": (100, 2), "E50": (100, 2), "E90": (100, 2)}
LABELS = {"citl": "CITL", "slope": "Calibration slope", "ICI": "ICI", "E50": "E50", "E90": "E90"}


def ipcw(t, e, horizon, counts):
    selected = counts > 0
    km = KaplanMeierFitter().fit(np.minimum(t[selected], horizon),
        event_observed=((e[selected] == 0) & (t[selected] < horizon)),
        weights=counts[selected])
    case = (e == 1) & (t <= horizon)
    control = t >= horizon
    assess = np.where(case, t, horizon)
    g = np.asarray(km.predict(np.nextafter(assess, -np.inf)))
    known = case | control
    if np.any(g[selected & known] <= 0):
        raise ValueError("Censoring support is zero at a required time.")
    weights = np.zeros(len(t))
    weights[known] = counts[known] / g[known]
    return case.astype(float), weights, float(km.predict(np.nextafter(float(horizon), -np.inf)))


def risk_basis(risk):
    p = np.clip(risk, 1e-6, 1 - 1e-6)
    z = logit(p)
    knots = np.quantile(z, [0.1, 0.5, 0.9])
    formula = f"cr(z, knots=({float(knots[1])},), lower_bound={float(knots[0])}, upper_bound={float(knots[2])}) - 1"
    return p, z, dmatrix(formula, dict(z=z)), knots


def diagnostic_fits(y, weights, z, basis, p, counts):
    use = weights > 0
    w, yy = weights[use], y[use]
    kwargs = dict(family=sm.families.Binomial(), freq_weights=w)
    citl = sm.GLM(yy, np.ones((use.sum(), 1)), offset=z[use], **kwargs).fit(maxiter=100)
    weak = sm.GLM(yy, sm.add_constant(z[use]), **kwargs).fit(maxiter=100)
    smooth = sm.GLM(yy, basis[use], **kwargs).fit(maxiter=100)
    if not all(f.converged for f in (citl, weak, smooth)):
        raise RuntimeError("A calibration diagnostic did not converge.")
    fitted = smooth.predict(basis)
    error = np.abs(fitted - p)
    repeated_error = np.repeat(error, counts.astype(int))
    metrics = [float(citl.params[0]), float(weak.params[0]), float(weak.params[1]),
               float(np.average(error, weights=counts)),
               float(np.quantile(repeated_error, 0.5)), float(np.quantile(repeated_error, 0.9))]
    return np.asarray(metrics), smooth


def analyze_horizon(d, site, horizon, n_boot, seed):
    t, e = d.PFS_TIME_DAYS.to_numpy(), d.PFS_EVENT.to_numpy()
    p, z, spline, knots = risk_basis(d[f"risk_{horizon}"].to_numpy())
    basis = np.asarray(spline)
    counts = np.ones(len(d), dtype=int)
    y, weights, g_horizon = ipcw(t, e, horizon, counts)
    point, smooth = diagnostic_fits(y, weights, z, basis, p, counts)
    grid = np.linspace(np.quantile(p, .01), np.quantile(p, .99), 150)
    grid_basis = np.asarray(build_design_matrices([spline.design_info], {"z": logit(grid)})[0])
    observed_grid = smooth.predict(grid_basis)
    group_codes, patient_ids = pd.factorize(d.PATIENT_ID, sort=True)
    rng = np.random.default_rng(seed)
    boot, curves, failures = [], [], []
    started = monotonic()
    for b in range(n_boot):
        cluster_counts = np.bincount(rng.integers(len(patient_ids), size=len(patient_ids)), minlength=len(patient_ids))
        multiplicity = cluster_counts[group_codes]
        try:
            yy, ww, _ = ipcw(t, e, horizon, multiplicity)
            with warnings.catch_warnings():
                warnings.simplefilter("error", sm.tools.sm_exceptions.PerfectSeparationWarning)
                values, fitted = diagnostic_fits(yy, ww, z, basis, p, multiplicity)
            boot.append(values)
            curves.append(fitted.predict(grid_basis))
        except (ValueError, RuntimeError, np.linalg.LinAlgError, sm.tools.sm_exceptions.PerfectSeparationWarning) as exc:
            failures.append(dict(cohort=site, horizon=horizon, replicate=b, error=str(exc)))
        if (b + 1) % 100 == 0:
            print(f"{site} {horizon} d: {b + 1}/{n_boot} patient bootstrap samples; {monotonic()-started:.1f} s.", flush=True)
    samples = np.asarray(boot)
    intervals = np.quantile(samples, [.025, .975], axis=0)
    row = dict(cohort=site, horizon=horizon, n_lines=len(d), n_patients=len(patient_ids),
        n_events_by_horizon=int(((e == 1) & (t <= horizon)).sum()),
        n_censored_before_horizon=int(((e == 0) & (t < horizon)).sum()),
        n_observed_event_free_through_horizon=int(((t >= horizon) & ~((e == 1) & (t <= horizon))).sum()),
        n_at_risk_just_before_horizon=int((t >= horizon).sum()),
        n_patients_at_risk_just_before_horizon=int(d.loc[t >= horizon, "PATIENT_ID"].nunique()),
        G_horizon_left=g_horizon, max_IPCW=float(weights.max()),
        weighted_effective_n=float(weights.sum() ** 2 / np.square(weights).sum()),
        mean_predicted_risk=float(p.mean()), IPCW_event_fraction=float(np.average(y, weights=weights)),
        n_bootstrap_requested=n_boot, n_bootstrap_success=len(boot),
        spline_knot_logit_10=knots[0], spline_knot_logit_50=knots[1], spline_knot_logit_90=knots[2])
    for j, metric in enumerate(METRICS):
        row[metric], row[f"{metric}_lo"], row[f"{metric}_hi"] = point[j], intervals[0, j], intervals[1, j]
    curve_intervals = np.quantile(curves, [.025, .975], axis=0)
    curve = pd.DataFrame(dict(cohort=site, horizon=horizon, predicted_risk=grid,
        observed_risk_smooth=observed_grid, smooth_lo=curve_intervals[0], smooth_hi=curve_intervals[1]))
    bootstrap = pd.DataFrame(samples, columns=METRICS)
    bootstrap.insert(0, "horizon", horizon)
    bootstrap.insert(0, "cohort", site)
    return row, curve, bootstrap, failures


def cell(r, key):
    scale, digits = SCALE[key]
    return f"{scale * r[key]:.{digits}f} [{scale * r[f'{key}_lo']:.{digits}f}, {scale * r[f'{key}_hi']:.{digits}f}]"


def calibration_table(metrics, cohort, labels=LABELS):
    d = metrics.loc[metrics.cohort.eq(cohort)].set_index("horizon")
    return pd.DataFrame([{"Measure": label, **{f"{h} days": cell(d.loc[h], key) for h in HORIZONS}}
                         for key, label in labels.items()])


def main():
    data = pd.read_csv(FROZEN)
    rows, curves, bootstraps = [], [], []
    for site in COHORTS:
        d = data.loc[data.cohort.eq(site)].reset_index(drop=True)
        for horizon in HORIZONS:
            row, curve, boot, failures = analyze_horizon(d, site, horizon, 1000, 20261009 + horizon)
            rows.append(row)
            curves.append(curve)
            bootstraps.append(boot)
            if failures:
                print(f"{site}, {horizon} days: {len(failures)} failed resamples.")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "curves.csv", index=False)
    pd.concat(bootstraps, ignore_index=True).to_csv(OUT / "bootstrap_metrics.csv", index=False)


if __name__ == "__main__":
    main()
