from __future__ import annotations

import numpy as np
from lifelines import KaplanMeierFitter


def km_curve(
    durations: np.ndarray,
    events: np.ndarray,
    *,
    label: str | None = None,
) -> tuple[np.ndarray, np.ndarray, KaplanMeierFitter]:
    """Fit a KM curve and return (times, survival, kmf)."""
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    if durations.size == 0:
        raise ValueError("durations must be non-empty.")
    kmf = KaplanMeierFitter()
    kmf.fit(durations=durations, event_observed=events, label=label)
    km_series = kmf.survival_function_[kmf._label]
    return (
        km_series.index.to_numpy(dtype=float),
        km_series.to_numpy(dtype=float),
        kmf,
    )


def rmst_from_curve(
    time_grid: np.ndarray,
    survival: np.ndarray,
    tau: float,
) -> float:
    """Compute RMST by integrating the survival curve up to tau."""
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        return float("nan")

    time_grid = np.asarray(time_grid, dtype=float)
    survival = np.asarray(survival, dtype=float)
    if time_grid.ndim != 1 or survival.ndim != 1:
        raise ValueError("time_grid and survival must be 1D sequences.")
    if time_grid.shape[0] != survival.shape[0]:
        raise ValueError("time_grid and survival must have the same length.")

    valid = np.isfinite(time_grid) & np.isfinite(survival)
    if not np.any(valid):
        return float("nan")
    time_grid = time_grid[valid]
    survival = survival[valid]
    order = np.argsort(time_grid)
    time_grid = time_grid[order]
    survival = survival[order]

    if time_grid[0] > 0:
        time_grid = np.concatenate([[0.0], time_grid])
        survival = np.concatenate([[survival[0]], survival])

    mask = time_grid <= tau
    times = time_grid[mask]
    surv_vals = survival[mask]
    if times.size == 0:
        return float(tau * survival[0])
    if times[-1] < tau:
        surv_at_tau = float(
            np.interp(tau, time_grid, survival, left=survival[0], right=survival[-1])
        )
        times = np.concatenate([times, [tau]])
        surv_vals = np.concatenate([surv_vals, [surv_at_tau]])
    return float(np.trapz(surv_vals, times))


def mae_pred_vs_km(
    time_grid: np.ndarray,
    pred_curve: np.ndarray,
    km_index: np.ndarray,
    km_values: np.ndarray,
) -> float:
    """Compute MAE between a predicted curve and a KM curve on the time grid."""
    km_index = np.asarray(km_index, dtype=float)
    km_values = np.asarray(km_values, dtype=float)
    if km_index.size == 0 or km_values.size == 0:
        return float("nan")
    pred_curve = np.asarray(pred_curve, dtype=float)
    km_interp = np.interp(
        time_grid,
        km_index,
        km_values,
        left=float(km_values[0]),
        right=float(km_values[-1]),
    )
    mask = np.isfinite(pred_curve) & np.isfinite(km_interp)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(np.abs(pred_curve[mask] - km_interp[mask])))
