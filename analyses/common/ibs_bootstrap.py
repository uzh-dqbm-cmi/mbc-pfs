from pathlib import Path

import numpy as np
import pandas as pd
from analyses.common.survival import SurvivalContext, ci, ibs_evaluator, load_curves, patient_draws

OUTPUT = Path("results/rebuttal/bootstrap")
COMPARISONS = [("full_gbsa", "ref_mlot1_ca15_3"),
               ("full_coxph", "ref_mlot1_ca15_3"),
               ("gbsa_no_line_missingness", "full_gbsa"),
               ("gbsa_site_count", "full_gbsa")]


def bootstrap_ibs(curves, grid, dm, weights=None):
    t, e = dm.PFS_TIME_DAYS.to_numpy(), dm.PFS_EVENT.to_numpy()
    pid, _, weights = patient_draws(dm, weights=weights)
    row_weights = np.vstack([np.ones(len(dm)), weights[:, pid]])
    times, time_index = np.unique(t, return_inverse=True)
    targets = np.unique(t[e == 0])
    target_index = np.searchsorted(times, targets)
    evaluator = ibs_evaluator(curves, grid, dm)
    probabilities = evaluator.predict_multi_probabilities_from_curve(targets)
    counts, censor_counts = [], []
    for row in row_weights:
        counts.append(np.bincount(time_index, weights=row, minlength=len(times)))
        censor_counts.append(np.bincount(time_index, weights=row * (1 - e), minlength=len(times)))
    counts, censor_counts = np.array(counts), np.array(censor_counts)
    at_risk = np.cumsum(counts[:, ::-1], axis=1)[:, ::-1]
    hazard = np.divide(censor_counts, at_risk, out=np.zeros_like(counts), where=at_risk > 0)
    censor_survival = np.cumprod(1 - hazard, axis=1)
    inverse = np.divide(1, censor_survival, out=np.zeros_like(counts), where=censor_survival > 0)
    event_error = np.square(probabilities) * (t[:, None] <= targets[None, :])
    alive_error = np.square(1 - probabilities) * (t[:, None] > targets[None, :])
    event_weights = row_weights * e * inverse[:, time_index]
    scores = ((event_error.T @ event_weights.T).T +
              (alive_error.T @ row_weights.T).T * inverse[:, target_index])
    scores /= row_weights.sum(axis=1)[:, None]
    values = []
    for row, present in zip(scores, censor_counts[:, target_index] > 0):
        support = targets[present]
        values.append(np.trapezoid(row[present], support) / (support[-1] - support[0]))
    return values[0], np.array(values[1:])


def paired(ctx, a, b):
    ibs = pd.read_csv(OUTPUT / "paired_IBS.csv").set_index("comparison").loc[f"{a} minus {b}"]
    return pd.Series(dict(ctx.compare(a, b), IBS_low=ibs.CI_low, IBS_high=ibs.CI_high))


def main():
    ctx = SurvivalContext()
    samples, points = {}, {}
    for name in dict.fromkeys(name for pair in COMPARISONS for name in pair):
        sf, grid = load_curves(ctx.dm, name)
        points[name], samples[name] = bootstrap_ibs(sf.to_numpy(), grid, ctx.dm, ctx.weights)
        print(f"{name}: IBS {points[name]:.9f}; completed {len(samples[name])} draws.", flush=True)
    rows = []
    for a, b in COMPARISONS:
        low, high = ci(samples[a] - samples[b])
        rows.append(dict(comparison=f"{a} minus {b}", IBS_a=points[a], IBS_b=points[b],
                         delta_IBS=points[a] - points[b], CI_low=low, CI_high=high,
                         n_bootstrap=len(samples[a]), seed=42))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(samples).to_csv(OUTPUT / "IBS_samples.csv", index=False)
    pd.DataFrame(rows).to_csv(OUTPUT / "paired_IBS.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
