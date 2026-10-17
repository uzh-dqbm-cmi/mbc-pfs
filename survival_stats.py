from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


def km_median_ci(
    durations: pd.Series,
    events: pd.Series,
    *,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    durations_num = pd.to_numeric(durations, errors="coerce").to_numpy(dtype=float)
    events_num = pd.to_numeric(events, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(durations_num) & np.isfinite(events_num)
    if not bool(np.any(mask)):
        return (np.nan, np.nan, np.nan)

    t = durations_num[mask]
    e = events_num[mask].astype(int)
    if t.size == 0:
        return (np.nan, np.nan, np.nan)

    order = np.argsort(t, kind="mergesort")
    t = t[order]
    e = e[order]

    event_times, d_counts = np.unique(t[e == 1], return_counts=True)
    if event_times.size == 0:
        return (float("inf"), float("inf"), float("inf"))

    n_total = t.size
    left_idx = np.searchsorted(t, event_times, side="left")
    n_at_risk = n_total - left_idx

    surv = np.empty(event_times.size, dtype=float)
    var_s = np.empty(event_times.size, dtype=float)
    greenwood_sum = 0.0
    s_prev = 1.0

    for i, (n_i, d_i) in enumerate(zip(n_at_risk, d_counts)):
        if n_i <= 0:
            surv[i] = s_prev
            var_s[i] = (s_prev * s_prev) * greenwood_sum
            continue

        frac = 1.0 - (d_i / n_i)
        s_prev = s_prev * frac
        surv[i] = s_prev

        if n_i > d_i and d_i > 0:
            greenwood_sum += d_i / (n_i * (n_i - d_i))
        var_s[i] = (s_prev * s_prev) * greenwood_sum

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    lower_s = np.empty_like(surv)
    upper_s = np.empty_like(surv)
    for i, (s, v) in enumerate(zip(surv, var_s)):
        if not np.isfinite(s) or s <= 0.0:
            lower_s[i] = 0.0
            upper_s[i] = 0.0
            continue
        if s >= 1.0:
            lower_s[i] = 1.0
            upper_s[i] = 1.0
            continue

        se = float(np.sqrt(max(v, 0.0)))
        denom = s * abs(np.log(s))
        if denom <= 0.0:
            lower_s[i] = np.nan
            upper_s[i] = np.nan
            continue
        se_g = se / denom
        g = float(np.log(-np.log(s)))

        upper_s[i] = float(np.exp(-np.exp(g - z * se_g)))
        lower_s[i] = float(np.exp(-np.exp(g + z * se_g)))

    def _first_time_leq(y: np.ndarray, threshold: float) -> float:
        valid_y = np.isfinite(y)
        if not bool(np.any(valid_y)):
            return np.nan
        idx = np.flatnonzero(valid_y & (y <= threshold))
        if idx.size == 0:
            return float("inf")
        return float(event_times[int(idx[0])])

    median = _first_time_leq(surv, 0.5)
    median_low = _first_time_leq(lower_s, 0.5)
    median_high = _first_time_leq(upper_s, 0.5)
    return (float(median), float(median_low), float(median_high))
