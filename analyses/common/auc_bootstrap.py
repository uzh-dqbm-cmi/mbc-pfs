import numba
import numpy as np
from sksurv.nonparametric import CensoringDistributionEstimator, SurvivalFunctionEstimator
from sksurv.util import Surv

from analyses.common.survival import patient_draws

METRICS = ("mean_auc", "auc_at_1y", "auc_at_2y")


@numba.njit
def weighted_auc(order, sorted_risk, times, duration, event, counts, ipcw, use):
    values = np.full(len(times), np.nan)
    for k in range(len(times)):
        if not use[k]:
            continue
        previous = 0.0
        started = False
        group_case = group_control = total_case = total_control = area = 0.0
        for j in range(len(duration)):
            i = order[k, j]
            if counts[i] == 0:
                continue
            risk = sorted_risk[k, j]
            if started and abs(risk - previous) > 1e-8:
                area += group_control * (total_case + .5 * group_case)
                total_case += group_case
                total_control += group_control
                group_case = group_control = 0.0
            if duration[i] <= times[k] and event[i]:
                group_case += counts[i] * ipcw[i]
            elif duration[i] > times[k]:
                group_control += counts[i]
            previous = risk
            started = True
        area += group_control * (total_case + .5 * group_case)
        total_case += group_case
        total_control += group_control
        values[k] = area / (total_case * total_control)
    return values


def evaluate(order, sorted_risk, grid, y, counts):
    active = counts > 0
    sampled_y = np.repeat(y, counts.astype(int))
    censoring = CensoringDistributionEstimator().fit(sampled_y)
    censoring_surv = censoring.predict_proba(grid)
    censoring_last = grid[censoring_surv > 0].max()
    event_times = sampled_y["time"][sampled_y["event"]]
    use = ((grid >= event_times.min()) & (grid <= event_times.max()) &
           (grid < sampled_y["time"].max()) & (grid <= censoring_last))
    ipcw = np.zeros(len(y))
    ipcw[active] = censoring.predict_ipcw(y[active])
    auc = weighted_auc(order, sorted_risk, grid, y["time"], y["event"], counts, ipcw, use)[use]
    times = grid[use]
    survival = SurvivalFunctionEstimator().fit(sampled_y).predict_proba(times)
    mean = np.sum(auc * -np.diff(np.r_[1., survival])) / (1 - survival[-1])
    return np.array([mean, auc[np.argmin(abs(times-365.))], auc[np.argmin(abs(times-730.))]])


def bootstrap_auc(curves, grid, dm, weights=None):
    pid, _, weights = patient_draws(dm, weights=weights)
    y = Surv.from_arrays(dm.PFS_EVENT.to_numpy(bool), dm.PFS_TIME_DAYS.to_numpy(np.float32))
    risks = 1.0 - np.asarray(curves)
    order = np.argsort(-risks, axis=0).T.astype(np.int32).copy()
    sorted_risk = np.take_along_axis(risks, order.T, axis=0).T.copy()
    point = evaluate(order, sorted_risk, grid, y, np.ones(len(dm), dtype=int))
    samples = np.empty((len(weights), len(METRICS)))
    for draw, cluster_counts in enumerate(weights):
        samples[draw] = evaluate(order, sorted_risk, grid, y, cluster_counts[pid].astype(int))
        if (draw+1) % 100 == 0:
            print(f"AUC: {draw+1}/{len(weights)} patient draws.", flush=True)
    return dict(zip(METRICS, point)), samples
