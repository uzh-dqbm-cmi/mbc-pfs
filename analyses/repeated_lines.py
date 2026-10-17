import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from scipy import stats
from tabulate import tabulate

from analyses.common.survival import SurvivalContext, aggregate, bootstrap_samples, ci

SELECTIONS = 30
SEED = 42


def leave_one_out(num, den):
    return ((num.sum() - num.sum(axis=0) - num.sum(axis=1) + np.diag(num)) /
            (den.sum() - den.sum(axis=0) - den.sum(axis=1) + np.diag(den)))


def off_diagonal(matrix):
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    return matrix


def pair_composition(result):
    num, den = result["num"], result["den"]
    within_num, within_den = np.trace(num), np.trace(den)
    return pd.DataFrame({"pairs": [den.sum(), within_den, den.sum() - within_den],
                         "C": [result["C"], within_num / within_den,
                               (num.sum() - within_num) / (den.sum() - within_den)]},
                        index=["all", "within", "between"])


def one_line_per_patient(ctx, result):
    dm = ctx.dm
    groups = list(dm.groupby("PATIENT_ID", sort=False).indices.values())
    t, e, curves = dm.PFS_TIME_DAYS.to_numpy(), dm.PFS_EVENT.to_numpy(), result["sf"].to_numpy()
    at_time = EvalSurv(result["sf"].T, t, e)._duration_idx()
    rng = np.random.default_rng(SEED)
    values, leave_out = [], []
    for _ in range(SELECTIONS):
        idx = np.array([rng.choice(group) for group in groups])
        num, den = aggregate(curves[idx], at_time[idx], t[idx], e[idx], ctx.pid[idx], len(groups), True)
        values.append(num.sum() / den.sum())
        leave_out.append(leave_one_out(num, den))
    values = np.array(values)
    return dict(patients=len(groups), mean=values.mean(), sd=values.std(ddof=1),
                leave_out=np.mean(leave_out, axis=0))


def paired_test(label, delta, leave_out):
    n = len(leave_out)
    se = np.sqrt((n - 1) / n * np.sum((leave_out - leave_out.mean()) ** 2))
    margin = stats.t.ppf(.975, n - 1) * se
    return [label, delta, delta - margin, delta + margin,
            2 * stats.t.sf(abs(delta / se), n - 1)]


def generate(context=None):
    ctx = context or SurvivalContext()
    dm = ctx.dm
    gbsa, cox = ctx.evaluate("full_gbsa"), ctx.evaluate("full_coxph")
    rows = []
    for label, result, seed in [("GBSA", gbsa, 46), ("CoxPH", cox, 43)]:
        rng = np.random.default_rng(seed)
        old_weights = np.array([np.bincount(rng.integers(len(ctx.ids), size=len(ctx.ids)),
                                           minlength=len(ctx.ids)) for _ in range(20)])
        old = ci(bootstrap_samples(result["num"], result["den"], ctx.self_events, old_weights))
        new = ci(result["samples"])
        rows.append([label, f'{result["C"]:.4f}', f"{old[0]:.3f}–{old[1]:.3f}",
                     f"{new[0]:.4f}–{new[1]:.4f}"])
    pairs = pair_composition(gbsa)
    pooled, between = pairs.C["all"], pairs.C["between"]
    pooled_leave = leave_one_out(gbsa["num"], gbsa["den"])
    between_leave = leave_one_out(off_diagonal(gbsa["num"]), off_diagonal(gbsa["den"]))
    single = one_line_per_patient(ctx, gbsa)
    tests = [paired_test("Between-patient minus pooled", between - pooled, between_leave - pooled_leave),
             paired_test("Mean one-line minus pooled", single["mean"] - pooled, single["leave_out"] - pooled_leave)]
    labels = {"all": "All pairs", "within": "Within-patient pairs", "between": "Between-patient pairs"}
    text = ["Repeated patient lines",
            f"Cohort: {len(dm):,} lines; {single['patients']:,} patients; {len(dm)/single['patients']:.2f} lines per patient.",
            "C uses pooled out-of-fold predictions. The updated intervals use 1,000 patient draws, seed 42.",
            tabulate(rows, headers=["Model", "Pooled C", "Original 95% CI (20 draws)", "Updated 95% CI"], tablefmt="psql", disable_numparse=True),
            tabulate([[labels[r.Index], f"{int(r.pairs):,}", f"{r.C:.4f}"] for r in pairs.itertuples()],
                     headers=["Pairs included", "Ordered pair contributions", "C"], tablefmt="psql", disable_numparse=True),
            tabulate([["All lines", f"{len(dm):,}", f"{pooled:.4f}"],
                      ["One random line per patient", f"{single['patients']:,}",
                       f"{single['mean']:.4f} ({single['sd']:.4f})"]],
                     headers=["Evaluation set", "Lines", f"C or mean (SD), {SELECTIONS} selections"], tablefmt="psql", disable_numparse=True),
            tabulate([[name, f"{delta:+.8f}", f"{lo:+.8f} to {hi:+.8f}", f"{p:.2e}"]
                      for name, delta, lo, hi, p in tests],
                     headers=["Comparison", "Change in C", "95% patient-jackknife CI", "Two-sided p"], tablefmt="psql", disable_numparse=True),
            "The jackknife removes each patient's lines together. All predictions stay fixed.",
            "Risk tertiles and SHAP contrasts use lines as rows."]
    return "\n\n".join(text) + "\n"


if __name__ == "__main__":
    print(generate())
