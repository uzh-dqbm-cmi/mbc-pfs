from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import restricted_mean_survival_time
from matplotlib import cycler
from matplotlib.lines import Line2D

from config import EVENT_COL, PATIENT_ID_COL, RANDOM_STATE, TIME_COL
from survival_metrics import km_curve, mae_pred_vs_km, rmst_from_curve

PROJECT_COLOR_CYCLE: list[str] = [
    "#1f4e79",
    "#d97706",
    "#166534",
    "#5499c7",
    "#f59e0b",
    "#22c55e",
]

SURVIVAL_X_LIMIT = (0, 730)
SURVIVAL_Y_LIMIT = (0, 1.02)

STYLE = {
    "figure.dpi": 200,
    "savefig.dpi": 400,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica:italic",
    "mathtext.bf": "Helvetica:bold",
    "mathtext.sf": "Helvetica",
    "mathtext.tt": "Helvetica",
    "mathtext.default": "regular",
    "mathtext.fallback": "stixsans",
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 12,
    "axes.labelweight": "regular",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#2f2f2f",
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.prop_cycle": cycler("color", PROJECT_COLOR_CYCLE),
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.color": "#2f2f2f",
    "ytick.color": "#2f2f2f",
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "lines.linewidth": 2.0,
}


STRATA_LABELS: tuple[str, str, str] = ("low", "mid", "high")
STRATA_COLORS: dict[str, str] = {
    "low": PROJECT_COLOR_CYCLE[0],
    "mid": PROJECT_COLOR_CYCLE[1],
    "high": PROJECT_COLOR_CYCLE[2],
}


def _ensure_png_path(output_path: Union[str, Path]) -> Path:
    path = Path(output_path)
    if path.suffix == "":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _assign_colors(
    keys: Sequence[str], palette: Sequence[str] = PROJECT_COLOR_CYCLE
) -> Dict[str, str]:
    seen: list[str] = []
    for key in keys:
        if key not in seen:
            seen.append(key)
    return {key: palette[i % len(palette)] for i, key in enumerate(seen)}


def plot_km_late_start_vs_rest(
    dm: pd.DataFrame,
    late_start_indices: Sequence[int],
    output_path: Union[str, Path],
    title: str = "KM curves: rest vs late-start agents",
    figsize: tuple[float, float] = (7.5, 4.5),
    dpi: int = 300,
) -> Path:
    plt.rcParams.update(STYLE)
    late_start_idx = np.asarray(late_start_indices, dtype=int)
    all_idx = np.arange(dm.shape[0], dtype=int)
    rest_idx = np.setdiff1d(all_idx, late_start_idx)

    def _km_curve(indices: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
        rows = dm.iloc[indices]
        durations = pd.to_numeric(rows[TIME_COL], errors="coerce").to_numpy(dtype=float)
        events = pd.to_numeric(rows[EVENT_COL], errors="coerce").to_numpy(dtype=int)
        mask = np.isfinite(durations) & np.isfinite(events)
        if not np.any(mask):
            raise ValueError(f"No valid KM data for {label}.")
        kmf = KaplanMeierFitter()
        kmf.fit(durations=durations[mask], event_observed=events[mask], label=label)
        km_series = kmf.survival_function_[kmf._label]
        return km_series.index.to_numpy(dtype=float), km_series.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    idx_rest, vals_rest = _km_curve(rest_idx, "Rest")
    idx_late, vals_late = _km_curve(late_start_idx, "Late start")
    ax.step(idx_rest, vals_rest, where="post", linewidth=2.0, label="Rest")
    ax.step(idx_late, vals_late, where="post", linewidth=2.0, label="Late start")
    ax.set_xlabel("Days since index")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target


def plot_km_modeled_vs_no_rad_prior(
    modeled_df: pd.DataFrame,
    no_rad_prior_df: pd.DataFrame,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
    title: str = "Kaplan–Meier PFS: modeled vs no-radiology-within-90d-prior",
    figsize: tuple[float, float] = (7.5, 4.5),
    dpi: int = 300,
    output_path: Union[str, Path] = Path("km_modeled_vs_no_rad_prior.png"),
) -> Path:
    plt.rcParams.update(STYLE)
    d0 = pd.to_numeric(modeled_df[time_col], errors="coerce").to_numpy(dtype=float)
    e0 = pd.to_numeric(modeled_df[event_col], errors="coerce").to_numpy(dtype=int)
    m0 = np.isfinite(d0) & np.isfinite(e0)
    d0, e0 = d0[m0], e0[m0]

    d1 = pd.to_numeric(
        no_rad_prior_df[f"ORIGINAL_{time_col}"], errors="coerce"
    ).to_numpy(dtype=float)
    e1 = (
        pd.to_numeric(no_rad_prior_df[f"ORIGINAL_{event_col}"], errors="coerce")
        == 1
    ).to_numpy(dtype=int)
    m1 = np.isfinite(d1)
    d1, e1 = d1[m1], e1[m1]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    kmf0 = KaplanMeierFitter()
    kmf0.fit(durations=d0, event_observed=e0, label="Modeled cohort")
    kmf0.plot_survival_function(
        ax=ax, ci_show=False, color=PROJECT_COLOR_CYCLE[0], lw=2.6
    )

    kmf1 = KaplanMeierFitter()
    kmf1.fit(durations=d1, event_observed=e1, label="Dropped: no radiology ≤90d prior")
    kmf1.plot_survival_function(
        ax=ax, ci_show=False, color=PROJECT_COLOR_CYCLE[1], lw=2.6, ls="--"
    )

    ax.set_title(title)
    ax.set_xlabel("Days since index")
    ax.set_ylabel("Progression-free survival probability")
    ax.set_xlim(SURVIVAL_X_LIMIT)
    ax.set_ylim(SURVIVAL_Y_LIMIT)
    ax.grid(True, which="major", alpha=0.35, linestyle="--", linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_mean_surv_late_start(
    dm: pd.DataFrame,
    time_grid: Sequence[float],
    surv_clean: pd.DataFrame,
    surv_leakage: pd.DataFrame,
    late_start_indices: Sequence[int],
    output_path: Union[str, Path],
    title: str = "Mean survival prediction: rest vs late-start agents",
    figsize: tuple[float, float] = (7.5, 4.5),
    dpi: int = 300,
) -> tuple[Path, Dict[str, float]]:
    plt.rcParams.update(STYLE)
    time_grid = np.asarray(time_grid, dtype=float).reshape(-1)
    late_start_idx = np.asarray(late_start_indices, dtype=int)
    all_idx = np.arange(surv_clean.shape[0], dtype=int)
    rest_idx = np.setdiff1d(all_idx, late_start_idx)

    c_rest_clean = PROJECT_COLOR_CYCLE[0]
    c_late_clean = PROJECT_COLOR_CYCLE[3]
    c_rest_leak = PROJECT_COLOR_CYCLE[1]
    c_late_leak = PROJECT_COLOR_CYCLE[4]
    c_km_rest = PROJECT_COLOR_CYCLE[2]
    c_km_late = PROJECT_COLOR_CYCLE[5]

    def _mean_surv_curve(surv_df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
        subset = surv_df.iloc[indices].to_numpy(dtype=float)
        return np.nanmean(subset, axis=0)

    rest_clean = _mean_surv_curve(surv_clean, rest_idx)
    rest_leakage = _mean_surv_curve(surv_leakage, rest_idx)
    late_clean = _mean_surv_curve(surv_clean, late_start_idx)
    late_leakage = _mean_surv_curve(surv_leakage, late_start_idx)

    def _curve_mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.nanmean(np.abs(a - b)))

    def _km_curve(indices: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
        rows = dm.iloc[indices]
        durations = pd.to_numeric(rows[TIME_COL], errors="coerce").to_numpy(dtype=float)
        events = pd.to_numeric(rows[EVENT_COL], errors="coerce").to_numpy(dtype=int)
        mask = np.isfinite(durations) & np.isfinite(events)
        if not np.any(mask):
            raise ValueError(f"No valid KM data for {label}.")
        kmf = KaplanMeierFitter()
        kmf.fit(durations=durations[mask], event_observed=events[mask], label=label)
        km_series = kmf.survival_function_[kmf._label]
        return km_series.index.to_numpy(dtype=float), km_series.to_numpy(dtype=float)

    km_time_rest, km_vals_rest = _km_curve(rest_idx, "Rest (KM)")
    km_time_late, km_vals_late = _km_curve(late_start_idx, "Late start (KM)")

    rest_clean_km = np.interp(km_time_rest, time_grid, rest_clean)
    rest_leakage_km = np.interp(km_time_rest, time_grid, rest_leakage)
    late_clean_km = np.interp(km_time_late, time_grid, late_clean)
    late_leakage_km = np.interp(km_time_late, time_grid, late_leakage)

    mae = {
        "rest_clean_vs_km": _curve_mae(rest_clean_km, km_vals_rest),
        "rest_leakage_vs_km": _curve_mae(rest_leakage_km, km_vals_rest),
        "late_clean_vs_km": _curve_mae(late_clean_km, km_vals_late),
        "late_leakage_vs_km": _curve_mae(late_leakage_km, km_vals_late),
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(
        time_grid,
        rest_clean,
        linewidth=2.4,
        color=c_rest_clean,
        label="Baseline prediction for rest of cohort",
    )
    ax.plot(
        time_grid,
        late_clean,
        linewidth=2.4,
        color=c_late_clean,
        label="Baseline prediction for lines with late-start agents",
    )
    ax.plot(
        time_grid,
        rest_leakage,
        linewidth=2.4,
        color=c_rest_leak,
        label="Leakage model prediction for rest of cohort",
    )
    ax.plot(
        time_grid,
        late_leakage,
        linewidth=2.4,
        color=c_late_leak,
        label="Leakage model prediction for lines with late-start agents",
    )
    ax.step(
        km_time_rest,
        km_vals_rest,
        where="post",
        linewidth=2.2,
        color=c_km_rest,
        ls="--",
        label="KM curve for rest of cohort",
    )
    ax.step(
        km_time_late,
        km_vals_late,
        where="post",
        linewidth=2.2,
        color=c_km_late,
        ls="--",
        label="KM curve for lines with late-start agents",
    )
    ax.set_xlabel("Days since index")
    ax.set_ylabel("Mean predicted survival")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target, mae


def plot_mean_surv_late_start_systemic_vs_local_clean_leakage(
    dm: pd.DataFrame,
    time_grid: Sequence[float],
    surv_clean: pd.DataFrame,
    surv_leakage: pd.DataFrame,
    late_start_systemic_indices: Sequence[int],
    late_start_local_indices: Sequence[int],
    output_path: Union[str, Path],
    *,
    title: str = "Mean survival vs KM: systemic vs local late-start (clean + leakage)",
    figsize: tuple[float, float] = (7.8, 4.8),
    dpi: int = 300,
) -> tuple[Path, Dict[str, float]]:
    plt.rcParams.update(STYLE)
    time_grid = np.asarray(time_grid, dtype=float).reshape(-1)
    systemic_idx = np.asarray(late_start_systemic_indices, dtype=int)
    local_idx = np.asarray(late_start_local_indices, dtype=int)

    if systemic_idx.size == 0 or local_idx.size == 0:
        raise ValueError("Both systemic and local indices must be non-empty.")

    c_systemic_clean = PROJECT_COLOR_CYCLE[0]
    c_local_clean = PROJECT_COLOR_CYCLE[3]
    c_systemic_leak = PROJECT_COLOR_CYCLE[1]
    c_local_leak = PROJECT_COLOR_CYCLE[4]
    c_km_systemic = PROJECT_COLOR_CYCLE[2]
    c_km_local = PROJECT_COLOR_CYCLE[5]

    def _mean_surv_curve(surv_df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
        subset = surv_df.iloc[indices].to_numpy(dtype=float)
        return np.nanmean(subset, axis=0)

    def _curve_mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.nanmean(np.abs(a - b)))

    def _km_curve(indices: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
        rows = dm.iloc[indices]
        durations = pd.to_numeric(rows[TIME_COL], errors="coerce").to_numpy(dtype=float)
        events = pd.to_numeric(rows[EVENT_COL], errors="coerce").to_numpy(dtype=int)
        mask = np.isfinite(durations) & np.isfinite(events)
        if not np.any(mask):
            raise ValueError(f"No valid KM data for {label}.")
        kmf = KaplanMeierFitter()
        kmf.fit(durations=durations[mask], event_observed=events[mask], label=label)
        km_series = kmf.survival_function_[kmf._label]
        return km_series.index.to_numpy(dtype=float), km_series.to_numpy(dtype=float)

    systemic_clean = _mean_surv_curve(surv_clean, systemic_idx)
    local_clean = _mean_surv_curve(surv_clean, local_idx)
    systemic_leak = _mean_surv_curve(surv_leakage, systemic_idx)
    local_leak = _mean_surv_curve(surv_leakage, local_idx)

    km_time_systemic, km_vals_systemic = _km_curve(systemic_idx, "Systemic (KM)")
    km_time_local, km_vals_local = _km_curve(local_idx, "Local (KM)")

    systemic_clean_km = np.interp(km_time_systemic, time_grid, systemic_clean)
    systemic_leak_km = np.interp(km_time_systemic, time_grid, systemic_leak)
    local_clean_km = np.interp(km_time_local, time_grid, local_clean)
    local_leak_km = np.interp(km_time_local, time_grid, local_leak)

    mae = {
        "systemic_clean_vs_km": _curve_mae(systemic_clean_km, km_vals_systemic),
        "systemic_leakage_vs_km": _curve_mae(systemic_leak_km, km_vals_systemic),
        "local_clean_vs_km": _curve_mae(local_clean_km, km_vals_local),
        "local_leakage_vs_km": _curve_mae(local_leak_km, km_vals_local),
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(
        time_grid,
        systemic_clean,
        linewidth=2.4,
        color=c_systemic_clean,
        label="Systemic (clean mean)",
    )
    ax.plot(
        time_grid,
        local_clean,
        linewidth=2.4,
        color=c_local_clean,
        label="Local (clean mean)",
    )
    ax.plot(
        time_grid,
        systemic_leak,
        linewidth=2.4,
        color=c_systemic_leak,
        label="Systemic (leakage mean)",
    )
    ax.plot(
        time_grid,
        local_leak,
        linewidth=2.4,
        color=c_local_leak,
        label="Local (leakage mean)",
    )
    ax.step(
        km_time_systemic,
        km_vals_systemic,
        where="post",
        linewidth=2.2,
        color=c_km_systemic,
        ls="--",
        label="Systemic (KM)",
    )
    ax.step(
        km_time_local,
        km_vals_local,
        where="post",
        linewidth=2.2,
        color=c_km_local,
        ls="--",
        label="Local (KM)",
    )

    ax.set_xlabel("Days since index")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target, mae


def _align_series(entry, time_points: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    times = np.asarray(entry.times, dtype=float).reshape(-1)
    order = np.argsort(times)
    times = times[order]
    mean = np.asarray(entry.mean, dtype=float).reshape(-1)[order]
    std = np.asarray(entry.std, dtype=float).reshape(-1)[order]
    keep = ~(np.isclose(times, 360.0) | np.isclose(times, 720.0))
    times, mean, std = times[keep], mean[keep], std[keep]
    points = np.asarray(time_points, dtype=float).reshape(-1)
    points = np.sort(points[~(np.isclose(points, 360.0) | np.isclose(points, 720.0))])
    idx = [int(np.argmin(np.abs(times - float(t)))) for t in points]
    return points, mean[idx], std[idx], int(entry.n)


def plot_full_group_individual_time_curves(
    models: Sequence[str],
    metric: str,
    time_points: Sequence[float],
    max_time_days: float,
    shade: str,
    shade_alpha: float,
    title: str,
    figsize: tuple[float, float],
    dpi: int,
    output_path: Union[str, Path],
    series: Mapping[str, Any],
    model_name_map: Mapping[str, str],
) -> Path:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    colors = _assign_colors(models, palette=PROJECT_COLOR_CYCLE)

    lo_vals: list[float] = []
    hi_vals: list[float] = []

    for name in models:
        entry = series[name]
        xs, mu, sd, runs = _align_series(entry, time_points)
        mask = xs <= float(max_time_days)
        xs, mu, sd = xs[mask], mu[mask], sd[mask]
        runs = max(runs, 1)
        half = 1.96 * sd / np.sqrt(float(runs)) if shade == "ci" and runs > 1 else sd
        ax.plot(
            xs,
            mu,
            marker="o",
            linewidth=2.0,
            label=model_name_map.get(name, name),
            color=colors[name],
        )
        ax.fill_between(xs, mu - half, mu + half, color=colors[name], alpha=shade_alpha)
        lo_vals.extend((mu - half).tolist())
        hi_vals.extend((mu + half).tolist())

    vmin = float(np.min(lo_vals))
    vmax = float(np.max(hi_vals))
    span = max(vmax - vmin, 1e-6)
    lo = vmin - 0.05 * span
    hi = vmax + 0.05 * span
    if metric in {"ipcw", "auc"}:
        lo = max(0.0, lo)
        hi = min(1.0, hi)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(metric)
    ax.set_ylim(lo, hi)
    ax.set_xticks(list(time_points))
    ax.grid(True, color="#cccccc", linewidth=0.6, alpha=0.4, linestyle="--")
    ax.set_title(title)
    ax.legend(loc="upper center", ncols=2)
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_hrher2_time_curves(
    metric: str,
    time_points: Sequence[float],
    shade: str,
    shade_alpha: float,
    y_limits: Tuple[float, float],
    title: str,
    figsize: tuple[float, float],
    dpi: int,
    output_path: Union[str, Path],
    subgroup_series: Mapping[str, Any],
) -> Path:
    plt.rcParams.update(STYLE)
    curves = [
        ("HR-/HER2-", "HR-_HER2-"),
        ("HR-/HER2+", "HR-_HER2+"),
        ("HR+/HER2-", "HR+_HER2-"),
        ("HR+/HER2+", "HR+_HER2+"),
    ]
    colors = _assign_colors([label for label, _ in curves], palette=PROJECT_COLOR_CYCLE)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for label, key in curves:
        if key not in subgroup_series:
            continue
        xs, mu, sd, runs = _align_series(subgroup_series[key], time_points)
        runs = max(runs, 1)
        half = 1.96 * sd / np.sqrt(float(runs)) if shade == "ci" and runs > 1 else sd
        ax.plot(xs, mu, marker="o", label=label, color=colors[label])
        ax.fill_between(xs, mu - half, mu + half, color=colors[label], alpha=shade_alpha)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(metric)
    ax.set_ylim(*y_limits)
    ax.set_xticks(list(time_points))
    ax.grid(True, color="#cccccc", linewidth=0.6, alpha=0.4, linestyle="--")
    ax.set_title(title)
    ax.legend(loc="upper center", ncols=2)
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_modality_best_model_metric_deltas(
    modality_perf: Mapping,
    baseline_stat,
    ablation_type: str,
    output_path: Path,
    figsize: tuple[float, float],
    dpi: int,
    model_name_map: Mapping[str, str] | None = None,
) -> Path:
    metrics = ["C", "mean_auc"]
    metric_labels = {"C": "C-index", "mean_auc": "Mean AUC"}

    modalities = sorted(modality_perf.keys())
    positions = np.arange(len(modalities), dtype=float)
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    extremes = []
    for j, metric in enumerate(metrics):
        heights = []
        errs = []
        offsets = positions + (j - (len(metrics) - 1) / 2.0) * width

        baseline_mean = float(baseline_stat.scalars[metric].mean)

        for modality in modalities:
            entry = modality_perf[modality][ablation_type]
            best_model = entry["best_model"]
            stats = entry[best_model]
            scalar = stats.scalars[metric]

            delta = float(scalar.mean) - baseline_mean
            heights.append(delta)
            errs.append(float(scalar.std))
            extremes.extend([delta - errs[-1], delta + errs[-1]])

        ax.bar(
            offsets, heights, width=width * 0.9, alpha=0.9, label=metric_labels[metric]
        )
        ax.errorbar(offsets, heights, yerr=errs, fmt="none", capsize=2.5, alpha=0.9)

    limit = max(abs(v) for v in extremes) if extremes else 0.05
    pad = max(0.01, 0.15 * limit)
    ax.set_ylim(-limit - pad, limit + pad)

    ax.axhline(0.0, linewidth=0.8, linestyle="--", alpha=0.8)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.4, linestyle="--")

    tick_labels = []
    for m in modalities:
        best_model = modality_perf[m][ablation_type]["best_model"]
        display_model = (
            model_name_map.get(best_model, best_model) if model_name_map else best_model
        )
        tick_labels.append(f"{m}\n({display_model})")
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=22, ha="right")

    ax.set_ylabel("Δ vs baseline (full model)")
    ax.set_title(f"{ablation_type.capitalize()} modality ablation (Δ)")
    ax.legend(loc="upper center", ncols=len(metrics))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_shap_summary_from_arrays(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: Sequence[str],
    label_map: Mapping[str, str] | None,
    top_k: int,
    title: str,
    figsize: tuple[float, float],
    dpi: int,
    output_path: Union[str, Path],
) -> Path:
    plt.rcParams.update(STYLE)
    shap_values = np.asarray(shap_values, dtype=float)
    X = np.asarray(X, dtype=float)
    if shap_values.ndim != 2 or X.ndim != 2:
        raise ValueError(
            "shap_values and X must be 2D arrays with shape (n_samples, n_features)."
        )
    if shap_values.shape != X.shape:
        raise ValueError("shap_values and X must have identical shapes.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")
    if len(feature_names) != shap_values.shape[1]:
        raise ValueError(
            "feature_names must align with shap_values/X columns: "
            f"{len(feature_names)} names vs {shap_values.shape[1]} columns."
        )
    display_names = [
        label_map.get(name, name) if label_map else name for name in feature_names
    ]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.sca(ax)
    rng_state = np.random.get_state()
    np.random.seed(int(RANDOM_STATE))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=display_names,
        max_display=top_k,
        show=False,
        plot_size=None,
    )
    np.random.set_state(rng_state)
    ax.set_title(title)
    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target


def plot_single_sample_shap_waterfall_from_arrays(
    shap_values: np.ndarray,
    X: np.ndarray,
    expected_value: float,
    feature_names: Sequence[str],
    label_map: Mapping[str, str],
    title: str,
    max_display: int,
    figsize: tuple[float, float],
    dpi: int,
    output_path: Union[str, Path],
) -> Path:
    plt.rcParams.update(STYLE)
    shap_values = np.asarray(shap_values, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float).reshape(-1)
    expected_value = float(np.asarray(expected_value).reshape(-1)[0])
    display_names = [label_map.get(name, name) for name in feature_names]
    explanation = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=X,
        feature_names=display_names,
    )
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    shap.plots.waterfall(
        explanation,
        max_display=max_display,
        show=False,
    )
    ax.set_title(title)
    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_group_mean_abs_shap(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    group_map: Mapping[str, str],
    label_map: Mapping[str, str] | None,
    top_k: int,
    title: str,
    figsize: tuple[float, float],
    dpi: int,
    output_path: Union[str, Path],
) -> Path:
    plt.rcParams.update(STYLE)
    shap_values = np.asarray(shap_values, dtype=float)
    feat_names = [str(f) for f in feature_names]
    if shap_values.ndim != 2:
        raise ValueError(
            f"Expected shap_values to be 2D (n_samples, n_features); got shape {shap_values.shape}."
        )
    if shap_values.shape[1] != len(feat_names):
        raise ValueError(
            "feature_names must align with shap_values columns: "
            f"{len(feat_names)} names vs {shap_values.shape[1]} SHAP columns."
        )

    group_to_indices: Dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(feat_names):
        group = group_map.get(name, "Other")
        if group == "Other":
            raise ValueError(f"Feature '{name}' missing from group_map.")
        group_to_indices[group].append(idx)

    group_mean: Dict[str, float] = {}
    n_samples = shap_values.shape[0]
    grouped_abs = np.zeros((n_samples, len(group_to_indices)), dtype=float)
    group_keys = list(group_to_indices.keys())

    for group_idx, group in enumerate(group_keys):
        indices = group_to_indices[group]
        group_sum = np.nansum(shap_values[:, indices], axis=1)
        grouped_abs[:, group_idx] = np.abs(group_sum)

    totals = np.nansum(grouped_abs, axis=1)
    shares = np.divide(
        grouped_abs,
        totals[:, None],
        out=np.zeros_like(grouped_abs),
        where=totals[:, None] > 0,
    )
    for group_idx, group in enumerate(group_keys):
        group_mean[group] = float(np.nanmean(shares[:, group_idx]))

    top = sorted(group_mean.items(), key=lambda x: x[1], reverse=True)[: int(top_k)]
    group_keys = [g for g, _ in top]
    labels = [label_map.get(g, g) if label_map else g for g in group_keys]
    values = [v for _, v in top]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    category_colors = {
        "Surveillance pattern": PROJECT_COLOR_CYCLE[0],
        "Biological intensity": PROJECT_COLOR_CYCLE[1],
        "Treatment": PROJECT_COLOR_CYCLE[2],
    }
    other_color = "#666666"
    pattern_keywords = {
        kw.lower()
        for kw in [
            "Measurement cadence",
            "Imaging coverage and cancer status",
        ]
    }
    bio_keywords = {
        kw.lower()
        for kw in [
            "Tumor marker kinetics",
            "Recent ECOG status",
            "Tumor site burden",
            "Genomics and therapeutic targets",
            "Initial diagnosis",
        ]
    }
    treatment_keywords = {
        kw.lower()
        for kw in [
            "Treatment exposure",
        ]
    }
    colors: list[str] = []
    categories: list[str] = []
    for key in group_keys:
        lower = key.lower()
        if any(kw in lower for kw in pattern_keywords):
            category = "Surveillance pattern"
        elif any(kw in lower for kw in bio_keywords):
            category = "Biological intensity"
        elif any(kw in lower for kw in treatment_keywords):
            category = "Treatment"
        else:
            print(f"Warning: unclassified group '{key}' for color assignment.")
            category = "Other"
        categories.append(category)
        colors.append(category_colors.get(category, other_color))

    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    xlabel = "Mean |Σ SHAP|"
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    legend_handles = []
    seen: set[str] = set()
    for category in ["Surveillance pattern", "Treatment", "Biological intensity"]:
        if category in categories and category not in seen:
            seen.add(category)
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=category_colors[category],
                    lw=6,
                    label=category.capitalize(),
                )
            )
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, loc="lower right")

    fig.tight_layout()
    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def _bootstrap_patient_metric(
    durations: np.ndarray,
    events: np.ndarray,
    patient_ids: Sequence[str],
    rmst_tau: float,
    n_boot: int = 400,
    random_state: int = RANDOM_STATE,
) -> tuple[float, tuple[float | None, float | None]]:
    rng = np.random.default_rng(random_state)
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    patient_ids = pd.Series(patient_ids, dtype=str)

    def _stat(d: np.ndarray, e: np.ndarray) -> float:
        kmf = KaplanMeierFitter()
        kmf.fit(durations=d, event_observed=e)
        return float(restricted_mean_survival_time(kmf, t=rmst_tau))

    point_estimate = _stat(durations, events)
    if n_boot <= 0:
        return point_estimate, (None, None)

    pid_groups = patient_ids.groupby(patient_ids, sort=False).indices
    unique_pids = np.asarray(list(pid_groups.keys()), dtype=object)
    if unique_pids.size == 0:
        return point_estimate, (None, None)
    pid_indices = [np.asarray(pid_groups[pid], dtype=int) for pid in unique_pids]

    samples: list[float] = []
    for _ in range(n_boot):
        sampled_idx = rng.integers(0, unique_pids.size, size=unique_pids.size)
        boot_indices = np.concatenate([pid_indices[i] for i in sampled_idx])
        if boot_indices.size == 0:
            continue
        samples.append(_stat(durations[boot_indices], events[boot_indices]))

    if not samples:
        return point_estimate, (None, None)
    low, high = np.percentile(samples, [2.5, 97.5])
    return point_estimate, (float(low), float(high))


def _normalize_rmst_taus(
    rmst_tau: float | None,
    rmst_taus: Sequence[float] | None,
) -> tuple[float, list[float]]:
    primary_tau = 365.0 if rmst_tau is None else float(rmst_tau)
    tau_list = [180.0, 365.0, 730.0] if rmst_taus is None else list(rmst_taus)
    if primary_tau not in tau_list:
        tau_list = [primary_tau] + tau_list

    cleaned: list[float] = []
    for tau in tau_list:
        try:
            tau_val = float(tau)
        except (TypeError, ValueError):
            continue
        if np.isfinite(tau_val) and tau_val > 0:
            if tau_val not in cleaned:
                cleaned.append(tau_val)
    if not cleaned:
        raise ValueError("rmst_taus must contain at least one positive value.")
    return primary_tau, cleaned


def _collect_tertile_stratified_survival_data(
    time_grid: Sequence[float],
    surv_np: np.ndarray,
    design_matrix: pd.DataFrame,
    strata_indices: Mapping[str, Sequence[int]],
    risk_scores: Sequence[float] | pd.Series,
    horizon_day: float,
    risk_table_times: Sequence[float],
    n_boot: int,
    random_state: int,
    rmst_tau: float | None,
    rmst_taus: Sequence[float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, object]], Dict[str, Any]]:
    time_grid = np.asarray(time_grid, dtype=float)
    surv_np = np.asarray(surv_np, dtype=float)
    if surv_np.ndim != 2 or time_grid.ndim != 1:
        raise ValueError("time_grid must match surv_np columns.")
    if surv_np.shape[1] != time_grid.shape[0]:
        raise ValueError("surv_np columns must match time_grid length.")
    horizon_day = float(horizon_day)
    matches = np.isclose(time_grid, horizon_day, atol=1e-6)
    if not np.any(matches):
        raise ValueError(
            "horizon_day must exist exactly in time_grid for pred horizon."
        )
    horizon_idx = int(np.flatnonzero(matches)[0])
    primary_rmst_tau, rmst_taus = _normalize_rmst_taus(rmst_tau, rmst_taus)

    if isinstance(risk_scores, pd.Series):
        risk_series = risk_scores.astype(float).reset_index(drop=True)
    else:
        risk_series = pd.Series(np.asarray(risk_scores, dtype=float))
    if risk_series.shape[0] != surv_np.shape[0]:
        raise ValueError("risk_scores length must match surv_np rows.")

    summary_rows: list[dict[str, object]] = []
    strata_indices_clean: Dict[str, list[int]] = {}
    lr_durations: list[np.ndarray] = []
    lr_events: list[np.ndarray] = []
    lr_groups: list[np.ndarray] = []
    strata_data: list[dict[str, Any]] = []
    rmst_rows: list[dict[str, object]] = []

    for stratum in STRATA_LABELS:
        member_idx = np.asarray(strata_indices.get(stratum, []), dtype=int)
        if member_idx.size == 0:
            continue
        mask = (member_idx >= 0) & (member_idx < surv_np.shape[0])
        if not np.any(mask):
            continue
        member_idx = np.unique(member_idx[mask])
        strata_indices_clean[stratum] = member_idx.tolist()

        curves = surv_np[member_idx]
        pred_mean = np.nanmean(curves, axis=0)
        if curves.shape[0] <= 1:
            pred_ci_low, pred_ci_high = pred_mean.copy(), pred_mean.copy()
        else:
            pred_ci_low, pred_ci_high = np.nanquantile(curves, 0.025, axis=0), np.nanquantile(curves, 0.975, axis=0)
        pred_ci_low = np.clip(pred_ci_low, 0.0, 1.0)
        pred_ci_high = np.clip(pred_ci_high, 0.0, 1.0)

        rows = design_matrix.iloc[member_idx]
        durations = pd.to_numeric(rows[TIME_COL], errors="coerce").to_numpy(dtype=float)
        events = pd.to_numeric(rows[EVENT_COL], errors="coerce").to_numpy(dtype=float)
        patient_ids = rows[PATIENT_ID_COL].astype(str).to_numpy()
        valid = np.isfinite(durations) & np.isfinite(events)
        durations = durations[valid]
        events = events[valid].astype(int)
        patient_ids = patient_ids[valid]

        rmst_val = float("nan")
        ci: tuple[float | None, float | None] = (None, None)
        km_index = None
        km_values = None
        rmst_km_by_tau = {tau: float("nan") for tau in rmst_taus}
        if durations.size > 0:
            km_index, km_values, kmf = km_curve(
                durations=durations,
                events=events,
                label=f"{stratum} observed KM",
            )
            for tau in rmst_taus:
                rmst_km_by_tau[tau] = float(restricted_mean_survival_time(kmf, t=tau))
            rmst_val, ci = _bootstrap_patient_metric(
                durations=durations,
                events=events,
                patient_ids=patient_ids,
                rmst_tau=primary_rmst_tau,
                n_boot=n_boot,
                random_state=random_state,
            )
            lr_durations.append(durations)
            lr_events.append(events)
            lr_groups.append(np.array([stratum] * durations.size))

        rmst_pred_by_tau = {
            tau: rmst_from_curve(time_grid, pred_mean, tau) for tau in rmst_taus
        }
        for tau in rmst_taus:
            rmst_km = rmst_km_by_tau.get(tau, float("nan"))
            rmst_pred = rmst_pred_by_tau.get(tau, float("nan"))
            rmst_rows.append(
                {
                    "stratum": stratum,
                    "tau": int(round(tau)),
                    "rmst_km": rmst_km,
                    "rmst_pred_mean": rmst_pred,
                    "rmst_diff": rmst_pred - rmst_km,
                }
            )

        risk_subset = risk_series.reindex(member_idx)
        risk_subset = risk_subset.dropna()
        risk_mean = float(risk_subset.mean()) if not risk_subset.empty else float("nan")
        risk_median = (
            float(risk_subset.median()) if not risk_subset.empty else float("nan")
        )
        summary_entry = {
            "stratum": stratum,
            "n": int(member_idx.size),
            "risk_mean": risk_mean,
            "risk_median": risk_median,
            "pred_surv_at_horizon": float(np.nanmean(surv_np[member_idx, horizon_idx])),
            "rmst": float(rmst_val) if np.isfinite(rmst_val) else np.nan,
            "rmst_ci_low": ci[0],
            "rmst_ci_high": ci[1],
            "rmst_pred_mean": rmst_pred_by_tau.get(primary_rmst_tau, float("nan")),
        }
        summary_rows.append(summary_entry)
        strata_data.append(
            {
                "stratum": stratum,
                "color": STRATA_COLORS[stratum],
                "pred_mean": pred_mean,
                "pred_ci_low": pred_ci_low,
                "pred_ci_high": pred_ci_high,
                "km_index": km_index,
                "km_values": km_values,
            }
        )

    if not summary_rows:
        raise ValueError("No strata with members were provided.")

    logrank_p: float | None = None
    if len(lr_groups) >= 2:
        try:
            durations_cat = np.concatenate(lr_durations)
            events_cat = np.concatenate(lr_events)
            groups_cat = np.concatenate(lr_groups)
            if (
                durations_cat.size == events_cat.size
                and np.unique(groups_cat).size >= 2
            ):
                lr_result = multivariate_logrank_test(
                    event_durations=durations_cat,
                    groups=groups_cat,
                    event_observed=events_cat.astype(int),
                )
                logrank_p = float(lr_result.p_value)
        except Exception as exc:
            print(f"Warning: log-rank test failed ({exc}).")
            logrank_p = None

    risk_table_times = [float(t) for t in risk_table_times]
    at_risk_rows: list[dict[str, object]] = []
    for stratum, idxs in strata_indices_clean.items():
        if not idxs:
            continue
        rows = design_matrix.iloc[idxs]
        durations = pd.to_numeric(rows[TIME_COL], errors="coerce").to_numpy(dtype=float)
        durations = durations[np.isfinite(durations)]
        row = {"stratum": stratum}
        for tp in risk_table_times:
            row[f"{int(tp)}d_at_risk"] = int(np.sum(durations >= tp))
        at_risk_rows.append(row)
    if at_risk_rows:
        at_risk_df = (
            pd.DataFrame(at_risk_rows)
            .set_index("stratum")
            .reindex(STRATA_LABELS)
            .reset_index()
        )
    else:
        at_risk_df = pd.DataFrame(columns=["stratum"])

    if rmst_rows:
        rmst_compare_df = pd.DataFrame(rmst_rows)
        rmst_compare_df["stratum"] = pd.Categorical(
            rmst_compare_df["stratum"], categories=STRATA_LABELS, ordered=True
        )
        rmst_compare_df = rmst_compare_df.sort_values(["stratum", "tau"]).reset_index(
            drop=True
        )
    else:
        rmst_compare_df = pd.DataFrame(
            columns=["stratum", "tau", "rmst_km", "rmst_pred_mean", "rmst_diff"]
        )

    details: Dict[str, Any] = {
        "strata_indices": strata_indices_clean,
        "horizon_day": horizon_day,
        "rmst_tau": primary_rmst_tau,
        "logrank_p": logrank_p,
        "at_risk_table": at_risk_df,
        "rmst_comparison": rmst_compare_df,
    }
    return strata_data, summary_rows, details


def plot_tertile_stratified_survival_and_km(
    time_grid: Sequence[float],
    surv_np: pd.DataFrame,
    design_matrix: pd.DataFrame,
    output_path: Union[str, Path],
    strata_indices: Mapping[str, Sequence[int]],
    risk_scores: pd.Series,
    horizon_day: float,
    rmst_tau: float,
    rmst_taus: Sequence[float],
    risk_table_times: Sequence[float],
    n_boot: int,
    random_state: int,
    subset_indices: Sequence[int] | None = None,
    cohort_label: str = "population",
    show_table: bool = True,
) -> tuple[Path, pd.DataFrame, Dict[str, Any]]:
    plt.rcParams.update(STYLE)
    if subset_indices is not None:
        subset_labels = list(dict.fromkeys(subset_indices))
        subset_idx = design_matrix.index.get_indexer(subset_labels)
        design_matrix = design_matrix.loc[subset_labels]
        surv_np = surv_np.loc[subset_labels]
        risk_scores = risk_scores.loc[subset_labels]
        strata_indices = _remap_strata_indices(strata_indices, subset_idx)

    strata_data, summary_rows, details = _collect_tertile_stratified_survival_data(
        time_grid=time_grid,
        surv_np=surv_np,
        design_matrix=design_matrix,
        strata_indices=strata_indices,
        risk_scores=risk_scores,
        horizon_day=horizon_day,
        rmst_tau=rmst_tau,
        rmst_taus=rmst_taus,
        risk_table_times=risk_table_times,
        n_boot=n_boot,
        random_state=random_state,
    )
    logrank_p = details.get("logrank_p")
    if logrank_p is not None:
        print(f"Log-rank p-value ({cohort_label}): {logrank_p:.3g}")

    time_grid = np.asarray(time_grid, dtype=float)
    surv_np = np.asarray(surv_np, dtype=float)
    mean_curve = np.nanmean(surv_np, axis=0)

    durations_all = pd.to_numeric(design_matrix[TIME_COL], errors="coerce").to_numpy(
        dtype=float
    )
    events_all = pd.to_numeric(design_matrix[EVENT_COL], errors="coerce").to_numpy(
        dtype=float
    )
    valid_all = np.isfinite(durations_all) & np.isfinite(events_all)
    durations_all = durations_all[valid_all]
    events_all = events_all[valid_all].astype(int)

    kmf_all = None
    if durations_all.size > 0:
        km_all_index, km_all_values, kmf_all = km_curve(
            durations=durations_all,
            events=events_all,
            label="Population observed KM",
        )
    else:
        km_all_index = np.array([], dtype=float)
        km_all_values = np.array([], dtype=float)

    if show_table:
        fig, (ax, ax_table) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8.6, 7.4),
            gridspec_kw={"height_ratios": [3.2, 1.3]},
        )
    else:
        fig, ax = plt.subplots(figsize=(8.6, 5.8))
        ax_table = None
    for row in strata_data:
        if row.get("pred_ci_low") is not None and row.get("pred_ci_high") is not None:
            ax.fill_between(
                time_grid,
                row["pred_ci_low"],
                row["pred_ci_high"],
                step="post",
                color=row["color"],
                alpha=0.12,
                linewidth=0.0,
            )
        ax.step(
            time_grid,
            row["pred_mean"],
            where="post",
            color=row["color"],
            lw=1.8,
            ls="--",
            alpha=0.8,
            label="_nolegend_",
        )
        if row["km_index"] is not None and row["km_values"] is not None:
            ax.step(
                row["km_index"],
                row["km_values"],
                where="post",
                color=row["color"],
                lw=2.2,
                ls="-",
                alpha=0.9,
                label="_nolegend_",
            )

    population_color = "#b91c1c"
    ax.step(
        time_grid,
        mean_curve,
        where="post",
        color=population_color,
        lw=2.4,
        ls="--",
        alpha=0.9,
        label="_nolegend_",
    )
    if km_all_index.size:
        ax.step(
            km_all_index,
            km_all_values,
            where="post",
            color=population_color,
            lw=2.8,
            ls="-",
            alpha=0.95,
            label="_nolegend_",
        )

    mae_by_label: dict[str, float] = {}
    mae_by_label["Population"] = mae_pred_vs_km(
        time_grid, mean_curve, km_all_index, km_all_values
    )
    for row in strata_data:
        if row["km_index"] is None or row["km_values"] is None:
            mae_by_label[row["stratum"]] = float("nan")
        else:
            mae_by_label[row["stratum"]] = mae_pred_vs_km(
                time_grid, row["pred_mean"], row["km_index"], row["km_values"]
            )

    def _format_mae(val: float) -> str:
        return "NA" if not np.isfinite(val) else f"{val:.3f}"

    ax.set_xlabel("Days since index")
    ax.set_ylabel("Survival probability")
    ax.set_xlim(SURVIVAL_X_LIMIT)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        f"Risk-stratified progression-free survival curves for {cohort_label} cohort",
        pad=8,
    )
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    stratum_label_map = {
        "low": "Low-risk",
        "mid": "Mid-risk",
        "high": "High-risk",
        "Population": "Population",
    }
    legend_handles = [
        Line2D(
            [],
            [],
            color="#111827",
            lw=2.0,
            ls="--",
            label="Prediction (dashed)",
        ),
        Line2D(
            [],
            [],
            color="#111827",
            lw=2.0,
            ls="-",
            label="Observed KM (solid)",
        ),
        Line2D(
            [],
            [],
            color=population_color,
            lw=2.6,
            label=f"Population (MAE={_format_mae(mae_by_label.get('Population', float('nan')))})",
        ),
        Line2D(
            [],
            [],
            color=STRATA_COLORS["low"],
            lw=2.4,
            label=(
                f"{stratum_label_map['low']} "
                f"(MAE={_format_mae(mae_by_label.get('low', float('nan')))})"
            ),
        ),
        Line2D(
            [],
            [],
            color=STRATA_COLORS["mid"],
            lw=2.4,
            label=(
                f"{stratum_label_map['mid']} "
                f"(MAE={_format_mae(mae_by_label.get('mid', float('nan')))})"
            ),
        ),
        Line2D(
            [],
            [],
            color=STRATA_COLORS["high"],
            lw=2.4,
            label=(
                f"{stratum_label_map['high']} "
                f"(MAE={_format_mae(mae_by_label.get('high', float('nan')))})"
            ),
        ),
    ]
    ax.legend(handles=legend_handles, frameon=False, ncol=2, loc="upper right")

    horizon_day_val = float(details.get("horizon_day", horizon_day))
    horizon_idx = int(np.argmin(np.abs(time_grid - horizon_day_val)))
    resolved_horizon = float(time_grid[horizon_idx])
    stats_lines = [f"Predicted S({int(resolved_horizon)}d)"]
    for stratum in STRATA_LABELS:
        idxs = details.get("strata_indices", {}).get(stratum, [])
        if not idxs:
            continue
        vals = surv_np[np.asarray(idxs, dtype=int), horizon_idx]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean_val = float(np.mean(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        label = stratum_label_map.get(stratum, str(stratum))
        stats_lines.append(f"{label}: {mean_val:.2f} [{min_val:.2f}-{max_val:.2f}]")
    if len(stats_lines) > 1:
        ax.text(
            0.02,
            0.02,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="#111827",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=3.0),
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.set_index("stratum").reindex(STRATA_LABELS).reset_index()
    summary_df["n"] = summary_df["n"].fillna(0).astype(int)
    summary_df["mae_pred_vs_km"] = summary_df["stratum"].map(mae_by_label)

    rmst_compare = details.get("rmst_comparison")

    combined_table = (
        summary_df[["stratum", "n"]]
        .merge(details["at_risk_table"], on="stratum", how="left")
        .rename(columns={"n": "count"})
    )
    rmst_tau_val = int(round(float(details.get("rmst_tau", rmst_tau))))
    if isinstance(rmst_compare, pd.DataFrame) and not rmst_compare.empty:
        rmst_tau_df = rmst_compare.loc[
            rmst_compare["tau"].astype(int) == rmst_tau_val,
            ["stratum", "rmst_km", "rmst_pred_mean", "rmst_diff"],
        ].rename(
            columns={
                "rmst_km": "RMST_KM",
                "rmst_pred_mean": "RMST prediction",
                "rmst_diff": "RMST delta",
            }
        )
        combined_table = combined_table.merge(rmst_tau_df, on="stratum", how="left")

    pop_row: dict[str, object] = {
        "stratum": "Population",
        "count": int(surv_np.shape[0]),
    }
    for tp in risk_table_times:
        pop_row[f"{int(tp)}d_at_risk"] = int(np.sum(durations_all >= float(tp)))
    if durations_all.size and kmf_all is not None:
        pop_row["RMST_KM"] = float(
            restricted_mean_survival_time(kmf_all, t=rmst_tau_val)
        )
    else:
        pop_row["RMST_KM"] = float("nan")
    pop_row["RMST prediction"] = rmst_from_curve(time_grid, mean_curve, rmst_tau_val)
    pop_row["RMST delta"] = float(pop_row["RMST prediction"]) - float(
        pop_row["RMST_KM"]
    )

    combined_table = pd.concat(
        [pd.DataFrame([pop_row]), combined_table], ignore_index=True
    )
    combined_table["stratum"] = combined_table["stratum"].map(
        lambda s: stratum_label_map.get(s, s)
    )
    combined_table = combined_table.drop(columns=["90d_at_risk"], errors="ignore")
    at_risk_cols = [
        c
        for c in combined_table.columns
        if re.fullmatch(r"\d+d_at_risk", str(c)) is not None
        and not str(c).startswith("90d")
    ]
    at_risk_cols = sorted(at_risk_cols, key=lambda c: int(str(c).split("d_at_risk")[0]))
    ordered_cols = (
        ["stratum", "count"]
        + at_risk_cols
        + [
            "RMST_KM",
            "RMST delta",
        ]
    )
    combined_table = combined_table[
        [c for c in ordered_cols if c in combined_table.columns]
    ]
    numeric_cols = [c for c in combined_table.columns if c != "stratum"]
    for col in numeric_cols:
        combined_table[col] = combined_table[col].round(0).astype(pd.Int64Dtype())

    display_cols = combined_table.columns.tolist()
    header_labels = {
        "stratum": "Stratum",
        "count": "Count",
        "RMST_KM": "KM RMST",
        "RMST delta": "\u0394RMST",
    }
    col_labels = [
        header_labels.get(col, col.replace("_at_risk", "").replace("d", " d"))
        for col in display_cols
    ]
    table_df = combined_table.copy()
    for col in display_cols:
        if col == "stratum":
            continue
        table_df[col] = table_df[col].apply(
            lambda x: "" if pd.isna(x) else f"{int(x):,}"
        )
    cell_text = table_df[display_cols].values.tolist()

    if show_table and ax_table is not None:
        ax_table.axis("off")
        table = ax_table.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="right",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12.5)
        table.scale(1.1, 1.6)
        last_row = max(row for row, _ in table.get_celld().keys())
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#111827")
            cell.set_linewidth(0.8)
            cell.visible_edges = ""
            if row_idx == 0:
                cell.set_facecolor("#f3f4f6")
                cell.visible_edges = "TB"
                if col_idx == 0:
                    cell.set_text_props(weight="bold", color="#111827", ha="left")
                else:
                    cell.set_text_props(weight="bold", color="#111827", ha="right")
            else:
                cell.set_facecolor("#fafafa" if row_idx % 2 == 1 else "white")
                if row_idx == last_row:
                    cell.visible_edges = "B"
                if col_idx == 0:
                    cell.set_text_props(ha="left")
                else:
                    cell.set_text_props(ha="right")
            cell.PAD = 0.28

        fig.subplots_adjust(hspace=0.18)
    out_path = _ensure_png_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)

    print("\nRisk-set counts (at risk) and RMST:")
    print(combined_table.to_string(index=False))

    return out_path, summary_df, details


def _remap_strata_indices(
    strata_indices: Mapping[str, Sequence[int]],
    subset_idx: Sequence[int],
) -> Dict[str, list[int]]:
    subset_idx_arr = np.asarray(subset_idx, dtype=int).ravel()
    ordered_idx = list(dict.fromkeys(int(idx) for idx in subset_idx_arr))
    mapping = {idx: new for new, idx in enumerate(ordered_idx)}
    remapped: Dict[str, list[int]] = {}
    for label, indices in strata_indices.items():
        dest = []
        for idx in indices:
            mapped = mapping.get(int(idx))
            if mapped is not None:
                dest.append(mapped)
        if dest:
            remapped[label] = dest
    return remapped


def plot_population_km_vs_mean(
    time_grid: Sequence[float],
    surv_np: np.ndarray,
    indices: Sequence[int],
    durations: Sequence[float],
    events: Sequence[int],
    output_path: Union[str, Path],
    title: str,
    figsize: tuple[float, float] = (8.0, 5.0),
    dpi: int = 400,
) -> tuple[Path, Dict[str, float]]:
    plt.rcParams.update(STYLE)
    time_grid = np.asarray(time_grid, dtype=float)
    surv_np = np.asarray(surv_np, dtype=float)
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        raise ValueError("indices cannot be empty for plotting.")
    if surv_np.shape[1] != time_grid.shape[0]:
        raise ValueError("surv_np columns must match time_grid length.")

    curves = surv_np[idx]
    if curves.ndim != 2:
        raise ValueError("surv_np[indices] must yield 2D array of curves.")
    mean_curve = np.nanmean(curves, axis=0)

    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    if durations.shape[0] != surv_np.shape[0] or events.shape[0] != surv_np.shape[0]:
        raise ValueError("durations/events must align with surv_np rows.")
    durations = durations[idx]
    events = events[idx]

    kmf = KaplanMeierFitter()
    kmf.fit(durations=durations, event_observed=events, label="Observed KM", alpha=0.05)
    observed_series = kmf.survival_function_at_times(time_grid)
    observed_curve = np.asarray(
        [float(observed_series.get(t, np.nan)) for t in time_grid],
        dtype=float,
    )
    ci = kmf.confidence_interval_.sort_index()
    ci_times = ci.index.to_numpy(dtype=float)
    ci_lower = ci.iloc[:, 0].to_numpy(dtype=float)
    ci_upper = ci.iloc[:, 1].to_numpy(dtype=float)

    n_total = int(durations.shape[0])
    at_risk_stats: dict[str, float] = {}
    for t in (180.0, 365.0, 730.0):
        n_at_risk = int(np.sum(durations >= t))
        pct_at_risk = (100.0 * n_at_risk / n_total) if n_total else float("nan")
        at_risk_stats[f"n_at_risk_{int(t)}"] = float(n_at_risk)
        at_risk_stats[f"pct_at_risk_{int(t)}"] = float(pct_at_risk)
    print(
        f"[{title}] N={n_total} | "
        f"At risk @180: {int(at_risk_stats['n_at_risk_180'])} ({at_risk_stats['pct_at_risk_180']:.1f}%) | "
        f"@365: {int(at_risk_stats['n_at_risk_365'])} ({at_risk_stats['pct_at_risk_365']:.1f}%) | "
        f"@730: {int(at_risk_stats['n_at_risk_730'])} ({at_risk_stats['pct_at_risk_730']:.1f}%)"
    )

    mask = np.isfinite(observed_curve) & np.isfinite(mean_curve)
    mae = (
        float(np.nanmean(np.abs(mean_curve[mask] - observed_curve[mask])))
        if np.any(mask)
        else float("nan")
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    mean_line = ax.step(
        time_grid,
        mean_curve,
        where="post",
        label="Mean predicted",
        color=PROJECT_COLOR_CYCLE[0],
        lw=3.0,
    )[0]
    ci_patch = ax.fill_between(
        ci_times,
        ci_lower,
        ci_upper,
        step="post",
        color=PROJECT_COLOR_CYCLE[1],
        alpha=0.2,
        label="KM 95% CI",
    )
    km_line = ax.step(
        observed_series.index,
        observed_series,
        where="post",
        label="Observed KM",
        color=PROJECT_COLOR_CYCLE[1],
        lw=2.6,
        ls="--",
    )[0]
    ax.set_xlabel("Days since index")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    risk_labels = [
        f"At risk @180d: {int(at_risk_stats['n_at_risk_180'])} ({at_risk_stats['pct_at_risk_180']:.1f}%)",
        f"At risk @365d: {int(at_risk_stats['n_at_risk_365'])} ({at_risk_stats['pct_at_risk_365']:.1f}%)",
        f"At risk @730d: {int(at_risk_stats['n_at_risk_730'])} ({at_risk_stats['pct_at_risk_730']:.1f}%)",
    ]
    risk_handles = [Line2D([], [], color="none", label=label) for label in risk_labels]
    legend_handles = [mean_line, km_line, ci_patch, *risk_handles]
    if np.isfinite(mae):
        legend_handles.append(
            Line2D([], [], color="none", label=f"MAE vs KM: {mae:.3f}")
        )
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right",
    )
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target, {"mae": mae, **at_risk_stats}
