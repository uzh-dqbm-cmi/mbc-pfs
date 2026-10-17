import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from scipy import stats

from analyses.common.survival import aggregate

SELECTIONS = 30
SEED = 42


def leave_one_out(num, den):
    return ((num.sum() - num.sum(axis=0) - num.sum(axis=1) + np.diag(num)) /
            (den.sum() - den.sum(axis=0) - den.sum(axis=1) + np.diag(den)))


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
