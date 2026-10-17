from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from tabulate import tabulate

from config import (ADMIN_CENSOR_DAYS, AUC, C_INDEX, DESIGN_MATRIX_PATH, FEATURES_DICT_PATH, IGNORE_COLS,
                    IGNORE_PREFIXES, MODALITY_GROUPS, RESULTS_PATH)

# %%
MODEL = "gbsa"
BASE_MODELS: Sequence[str] = sorted(("coxph", "deephit", "deepsurv", "gbsa", "rsf"))
DEEPHIT_BOOTSTRAP_INTERP_GRID = np.arange(0.0, float(ADMIN_CENSOR_DAYS), 1.0)
DEFAULT_MODEL_NAME_MAP = {"coxph": "CoxPH", "deephit": "DeepHit", "deepsurv": "DeepSurv", "gbsa": "GBSA", "rsf": "RSF"}
SUBTYPES = {"HR-/HER2-": (0, 0), "HR-/HER2+": (0, 1), "HR+/HER2-": (1, 0), "HR+/HER2+": (1, 1)}
SUBTYPE_ORDER = tuple(SUBTYPES)
MODALITIES = [modality for modality in MODALITY_GROUPS if modality != "Surveillance Patterns"]
MANUSCRIPT_TIME_HORIZONS: tuple[int, ...] = (90, 180, 365, 730)
MANUSCRIPT_INTERMEDIATE_RISK_DAYS = 365
MANUSCRIPT_BASELINE_CONTRAST_EXCLUDED_FEATURES: tuple[str, ...] = ("IS_MLOT1", "GENOMICS_MISSING")
EXCLUDED_TREATMENT_LABELS = {"SURGERY", "RADIATION_THERAPY", "BONE TREATMENT"}
FEATURE_GROUPS_PATH = Path("data/feature_groups.json")
DELTA_METRICS = {"delta_C": C_INDEX, "delta_IBS": "IBS", "delta_AUC": AUC}
RULE_CLAUSES = [("include_exact", "exact feature", "exact features"), ("include_prefix", "prefix", "prefixes"),
                ("include_regex", "regex", "regex"), ("exclude_exact", "exclude exact feature", "exclude exact features"),
                ("exclude_prefix", "exclude prefix", "exclude prefixes"), ("exclude_regex", "exclude regex", "exclude regex")]


def assign_tertiles(values: pd.Series) -> pd.Series:
    """Assign `low`/`mid`/`high` by equal-frequency thirds of the valid scores."""
    values = pd.to_numeric(values, errors="coerce")
    valid = values.dropna()
    strata = pd.Series(index=values.index, dtype=object)
    if valid.empty:
        return strata
    ranks = valid.rank(method="first")
    n = ranks.shape[0]
    cutoff1 = int(np.ceil(n / 3))
    cutoff2 = int(np.ceil(2 * n / 3))
    ordered_idx = ranks.sort_values().index
    strata.loc[ordered_idx[:cutoff1]] = "low"
    strata.loc[ordered_idx[cutoff1:cutoff2]] = "mid"
    strata.loc[ordered_idx[cutoff2:]] = "high"
    return strata


def tertile_strata(scores: pd.Series, subset_indices: Sequence[int] | None = None) -> dict[str, list[int]]:
    strata = assign_tertiles(scores if subset_indices is None else scores.loc[list(subset_indices)])
    return {label: strata[strata == label].index.to_list() for label in ("low", "mid", "high")
            if (strata == label).any()}


@dataclass
class ScalarMetricSummary:
    mean: float
    std: float
    values: list[float]


@dataclass
class TimeMetricSeries:
    times: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n: int


@dataclass
class ModelStat:
    name: str
    scalars: Dict[str, ScalarMetricSummary]
    series: Dict[str, TimeMetricSeries]
    hrher2_series: Dict[str, Dict[str, TimeMetricSeries]]


def median_pfs(pfs: pd.Series, events: pd.Series) -> float:
    return median_pfs_ci(pfs, events)[0]


def median_pfs_ci(pfs: pd.Series, events: pd.Series, alpha: float = 0.05) -> tuple[float, float, float]:
    # Kaplan–Meier median PFS (accounts for censoring).
    from survival_stats import km_median_ci

    return km_median_ci(pfs, events, alpha=alpha)


def format_median_ci(median: float, lower: float, upper: float, *, digits: int = 2) -> str:
    if np.isnan(median):
        return "NA"
    if np.isinf(median):
        return "NR"
    fmt = f"{{:.{digits}f}}"
    median_str = fmt.format(median)
    if np.isfinite(lower) and np.isfinite(upper):
        return f"{median_str} ({fmt.format(lower)}-{fmt.format(upper)})"
    return f"{median_str} (NA)"


def subtype_label(hr, her2) -> str:
    return f"HR{'+' if hr else '-'}/HER2{'+' if her2 else '-'}"


def subtype_masks(dm: pd.DataFrame) -> dict[str, pd.Series]:
    return {name: (dm["HR"] == hr) & (dm["HER2"] == her2) for name, (hr, her2) in SUBTYPES.items()}


def pretty_combo_label(col: str, prefix: str) -> str:
    label = col
    prefix_token = f"{prefix}_"
    if label.startswith(prefix_token):
        label = label[len(prefix_token):]
    return re.sub(r"_+", " ", label).strip().title()


def is_excluded_treatment_label(label: str) -> bool:
    normalized = re.sub(r"[_\s]+", " ", label).strip().upper()
    return any(f" {excluded} " in f" {normalized} " for excluded in EXCLUDED_TREATMENT_LABELS)


def aggregate_agent_treatment_categories(dm: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    agent_prefix = "PLANNED_AGENT_"
    agent_cols = [col for col in dm.columns if col.startswith(agent_prefix) and not is_excluded_treatment_label(col)]
    if not agent_cols:
        return pd.DataFrame()
    groups: dict[str, list[str]] = {}
    for col in agent_cols:
        remainder = col[len(agent_prefix):]
        if "_" not in remainder:
            continue
        category = remainder.rsplit("_", 1)[0]
        if is_excluded_treatment_label(category):
            continue
        groups.setdefault(category, []).append(col)
    if not groups:
        return pd.DataFrame()
    totals = {category: sum((float(dm[c].astype(float).sum()) for c in cols), 0.0) for category, cols in groups.items()}
    sorted_cats = sorted(totals, key=lambda key: totals[key], reverse=True)
    top_cats = sorted_cats[:top_n]
    aggregated: dict[str, pd.Series] = {}
    for category in top_cats:
        aggregated[category] = (dm[groups[category]].fillna(0).astype(float).sum(axis=1) >= 1).astype(int)
    leftover = [cat for cat in sorted_cats if cat not in top_cats]
    if leftover:
        other_cols = [col for cat in leftover for col in groups[cat]]
        aggregated["Other"] = (dm[other_cols].fillna(0).astype(float).sum(axis=1) >= 1).astype(int)
    sanitized: dict[str, pd.Series] = {}
    for category, series in aggregated.items():
        safe = re.sub(r"[^A-Z0-9]+", "_", category.upper()).strip("_") or "OTHER"
        sanitized[f"PLANNED_TREATMENT_{safe}"] = series.astype(int)
    return pd.DataFrame(sanitized, index=dm.index)


def treatment_combination_counts(dm: pd.DataFrame, col_name: str, top_n: int = 5) -> pd.DataFrame:
    prefix = f"PLANNED_{col_name}"
    if col_name == "TREATMENT":
        flags = aggregate_agent_treatment_categories(dm, top_n=top_n)
    else:
        flags = dm[[c for c in dm.columns if c.startswith(prefix) and not is_excluded_treatment_label(c)]].fillna(0)
    flags = flags.astype(int)
    columns = list(flags.columns)
    table = pd.concat([dm[["HR", "HER2"]].reset_index(drop=True), flags.reset_index(drop=True)], axis=1)
    counts = (table.groupby(["HR", "HER2"] + columns).size().reset_index(name="count")
              .sort_values(by=["HR", "HER2", "count"], ascending=[True, True, False]))
    counts["percentage"] = counts["count"] / counts.groupby(["HR", "HER2"])["count"].transform("sum") * 100
    counts["regimen"] = counts[columns].apply(
        lambda row: ", ".join(pretty_combo_label(c, prefix) for c, v in row.items() if v == 1) or "None", axis=1)
    return counts[["HR", "HER2", "count", "percentage", "regimen"]]


def top_treatment_combinations(dm: pd.DataFrame, col_name: str, top_k: int) -> pd.DataFrame:
    top = treatment_combination_counts(dm, col_name).groupby(["HR", "HER2"]).head(top_k)
    return top.assign(subtype=[subtype_label(hr, her2) for hr, her2 in zip(top.HR, top.HER2)],
                      rank=top.groupby(["HR", "HER2"]).cumcount() + 1).reset_index(drop=True)


def treatment_combinations(dm: pd.DataFrame, col_name: str, *, top_n: int = 5) -> None:
    counts = treatment_combination_counts(dm, col_name.upper(), top_n=top_n)
    for (hr, her2), group in counts.groupby(["HR", "HER2"]):
        print(f"\nTop treatments for {subtype_label(hr, her2)}:")
        print(tabulate(group.head(5)[["regimen", "count", "percentage"]], showindex=False))


def overall_performance_metric_rows() -> list[tuple[str, str, bool]]:
    return [("C-index", C_INDEX, False), ("mean AUC", AUC, False), ("AUC @ 1y", "auc_at_1y", False),
            ("AUC @ 2y", "auc_at_2y", False), ("IBS", "IBS", True)]


def overall_performance(model_perf: Mapping[str, ModelStat], models: Sequence[str] = BASE_MODELS) -> pd.DataFrame:
    rows = overall_performance_metric_rows()
    stat = lambda attr: pd.DataFrame([[getattr(model_perf[model].scalars[key], attr) for model in models]
                                      for _, key, _ in rows], index=[label for label, _, _ in rows], columns=list(models))
    return pd.concat({"mean": stat("mean"), "std": stat("std")}, axis=1)


def resolve_time_index(times: np.ndarray, horizon: float) -> int:
    times = np.asarray(times, dtype=float)
    exact = np.where(times == float(horizon))[0]
    if exact.size:
        return int(exact[0])
    return int(np.argmin(np.abs(times - float(horizon))))


def horizon_summary(series_by_name: Mapping[str, TimeMetricSeries],
                    horizons: Sequence[int] = MANUSCRIPT_TIME_HORIZONS) -> pd.DataFrame:
    index = {name: [resolve_time_index(series.times, h) for h in horizons] for name, series in series_by_name.items()}
    stat = lambda attr: pd.DataFrame([getattr(series, attr)[index[name]] for name, series in series_by_name.items()],
                                     index=list(series_by_name), columns=list(horizons))
    return pd.concat({"mean": stat("mean"), "std": stat("std")}, axis=1)


def mean_sd(stats: pd.DataFrame, sep: str = " ± ") -> pd.DataFrame:
    return stats["mean"].map("{:.3f}".format) + sep + stats["std"].map("{:.3f}".format)


def _interp_survival_matrix_to_grid(
    time_grid: np.ndarray,
    surv_np: np.ndarray,
    target_time_grid: np.ndarray,
) -> np.ndarray:
    time_grid = np.asarray(time_grid, dtype=float).reshape(-1)
    surv_np = np.asarray(surv_np, dtype=float)
    target_time_grid = np.asarray(target_time_grid, dtype=float).reshape(-1)
    if surv_np.ndim != 2 or surv_np.shape[1] != time_grid.shape[0]:
        raise ValueError("surv_np must be a (n_samples, n_times) array aligned with time_grid.")
    if time_grid.size == 0 or target_time_grid.size == 0:
        raise ValueError("time grids must be non-empty.")
    if np.any(np.diff(time_grid) < 0):
        raise ValueError("time_grid must be sorted in non-decreasing order.")
    interp_rows = [np.interp(target_time_grid, time_grid, row, left=float(row[0]), right=float(row[-1]))
                   for row in surv_np]
    return np.asarray(interp_rows, dtype=float)


def _build_time_series_from_folds(fold_metrics: Sequence[Mapping], metric_key: str) -> TimeMetricSeries | None:
    arrays: list[np.ndarray] = []
    times = None
    for fm in fold_metrics:
        if metric_key not in fm or "eval_horizons" not in fm:
            continue
        if times is None:
            times = np.asarray(fm["eval_horizons"], dtype=float)
        arrays.append(np.asarray(fm[metric_key], dtype=float))
    if not arrays or times is None:
        return None
    stacked = np.vstack(arrays)
    drop_times = []
    if 365.0 in times and 360.0 in times:
        drop_times.append(360.0)
    if 730.0 in times and 720.0 in times:
        drop_times.append(720.0)
    if drop_times:
        keep_mask = ~np.isin(times, np.asarray(drop_times, dtype=float))
        times = times[keep_mask]
        stacked = stacked[:, keep_mask]
    return TimeMetricSeries(times=times, mean=stacked.mean(axis=0), std=stacked.std(axis=0), n=stacked.shape[0])


def _build_hrher2_series(fold_metrics: Sequence[Mapping]) -> Dict[str, Dict[str, TimeMetricSeries]]:
    accum: dict[str, dict[str, list[np.ndarray]]] = {"auc": {}, "ipcw": {}}
    sample_times: dict[str, dict[str, np.ndarray]] = {"auc": {}, "ipcw": {}}
    for fm in fold_metrics:
        for subgroup, stats in fm.get("hr_her2_metrics", {}).items():
            times = np.asarray(stats.get("eval_times", []), dtype=float)
            if times.size == 0:
                continue
            for metric_key in ("auc", "ipcw"):
                values = stats.get(metric_key)
                if values is None:
                    continue
                arr = np.asarray(values, dtype=float)
                if arr.size != times.size:
                    continue
                sample_times[metric_key].setdefault(subgroup, times)
                accum[metric_key].setdefault(subgroup, []).append(arr)
    hrher_map: Dict[str, Dict[str, TimeMetricSeries]] = {}
    for metric_key in ("auc", "ipcw"):
        sub_map: Dict[str, TimeMetricSeries] = {}
        for subgroup, arrays in accum[metric_key].items():
            if not arrays:
                continue
            stacked = np.vstack(arrays)
            sub_map[subgroup] = TimeMetricSeries(times=sample_times[metric_key][subgroup], mean=stacked.mean(axis=0),
                                                 std=stacked.std(axis=0), n=stacked.shape[0])
        if sub_map:
            hrher_map[metric_key] = sub_map
    return hrher_map


def build_model_perf(results_root: Path, base_models: Sequence[str]) -> tuple[Dict[str, ModelStat], Dict[str, list[Path]]]:
    model_perf: Dict[str, ModelStat] = {}
    model_results_paths: Dict[str, list[Path]] = {}
    for model in base_models:
        folds = list(json.loads((Path(results_root) / model / "eval_metrics.json").read_text())["outer_folds"].values())
        model_results_paths[model] = [Path(fold["results_path"]) for fold in folds]
        metrics = [fold["metrics"] for fold in folds]
        values = {C_INDEX: [fm[C_INDEX] for fm in metrics], AUC: [fm[AUC] for fm in metrics],
                  "IBS": [fm["IBS"] for fm in metrics],
                  "auc_at_1y": [float(fm["auc"][int(np.argmin(np.abs(np.asarray(fm["eval_horizons"], dtype=float) - 365.0)))])
                                for fm in metrics],
                  "auc_at_2y": [float(fm["auc"][-1]) for fm in metrics]}
        scalars = {name: ScalarMetricSummary(mean=float(np.mean(v)), std=float(np.std(v)), values=[float(x) for x in v])
                   for name, v in values.items()}
        series = {key: ts for key in ("auc", "ipcw", "brier_score") if (ts := _build_time_series_from_folds(metrics, key))}
        model_perf[model] = ModelStat(name=model, scalars=scalars, series=series,
                                      hrher2_series=_build_hrher2_series(metrics))
    return model_perf, model_results_paths


def load_oof_surv_only(dm, surv_paths, target_time_grid: np.ndarray | None = None):
    # --- build common time grid ---
    if target_time_grid is None:
        time_list = []
        for p in surv_paths:
            with np.load(p) as f:
                t = np.asarray(f["time"], dtype=float)
            time_list.append(np.unique(t))
        common_time = time_list[0]
        for t in time_list[1:]:
            common_time = np.intersect1d(common_time, t, assume_unique=True)
        if common_time.size == 0:
            raise ValueError("No overlapping time points across folds.")
    else:
        common_time = np.asarray(target_time_grid, dtype=float).reshape(-1)
        if common_time.size == 0:
            raise ValueError("target_time_grid must be non-empty.")
        common_time = np.unique(common_time)

    # --- initialize arrays ---
    n = len(dm)
    surv_oof = np.full((n, common_time.size), np.nan, dtype=float)

    # --- fill OOF arrays ---
    for surv_path in surv_paths:
        with np.load(surv_path) as sf:
            idx_test = np.asarray(sf["idx_test"], dtype=int)
            time_grid = np.asarray(sf["time"], dtype=float)
            surv_vals = np.asarray(sf["surv"], dtype=float)
            if target_time_grid is None:
                # align columns to the common grid
                col_idx = np.searchsorted(time_grid, common_time)
                if not np.allclose(time_grid[col_idx], common_time):
                    raise ValueError(f"Time grid mismatch in {surv_path}")
                surv_oof[idx_test] = surv_vals[:, col_idx]
            else:
                surv_oof[idx_test] = _interp_survival_matrix_to_grid(
                    time_grid=time_grid,
                    surv_np=surv_vals,
                    target_time_grid=common_time,
                )

    # --- wrap as DataFrame for downstream use ---
    surv_df = pd.DataFrame(surv_oof, index=dm.index, columns=common_time)
    return surv_df, common_time


def load_oof_surv_and_shap(dm, surv_paths, shap_paths):
    surv_df, common_time = load_oof_surv_only(dm, surv_paths)
    n = len(dm)
    # load feature names from any shap file
    with np.load(shap_paths[0], allow_pickle=True) as f:
        feature_names = np.asarray(f["feature_names"]).astype(str)
    shap_oof = np.full((n, feature_names.size), np.nan, dtype=float)
    shap_X = np.full((n, feature_names.size), np.nan, dtype=float)
    shap_expected = np.full(n, np.nan, dtype=float)
    # --- fill OOF arrays ---
    for surv_path, shap_path in zip(surv_paths, shap_paths):
        with np.load(surv_path) as sf:
            idx_test = np.asarray(sf["idx_test"], dtype=int)
        with np.load(shap_path, allow_pickle=True) as hf:
            sample_idx = np.asarray(hf["sample_idx"], dtype=int)
            if not np.array_equal(np.sort(sample_idx), np.sort(idx_test)):
                raise ValueError(f"Index mismatch between {surv_path} and {shap_path}")
            shap_oof[sample_idx] = np.asarray(hf["shap_values"], dtype=float)
            shap_X[sample_idx] = np.asarray(hf["X"], dtype=float)
            shap_expected[sample_idx] = np.asarray(hf["expected_value"]).reshape(-1)
    # --- wrap as DataFrames/Series for downstream use ---
    shap_df = pd.DataFrame(shap_oof, index=dm.index, columns=feature_names)
    shap_X_df = pd.DataFrame(shap_X, index=dm.index, columns=feature_names)
    shap_expected_s = pd.Series(shap_expected, index=dm.index, name="expected_value")
    return surv_df, shap_df, shap_X_df, shap_expected_s, common_time, feature_names


def paired_signflip_test(x: Sequence[float], y: Sequence[float], *, alternative: str = "two-sided") -> Dict[str, float]:
    """Exact paired sign-flip test on outer-fold differences (good for n=5)."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape for paired testing.")
    d = y_arr - x_arr
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n == 0:
        return {"n": 0.0, "mean_diff": float("nan"), "p_value": float("nan")}
    obs = float(np.mean(d))
    # enumerate all 2^n sign flips (n is tiny for nested-CV outer folds)
    flips = np.asarray([((mask >> np.arange(n)) & 1) * 2 - 1 for mask in range(1 << n)], dtype=float)
    means = flips.dot(d) / float(n)
    if alternative == "greater":
        p = float(np.mean(means >= obs))
    elif alternative == "less":
        p = float(np.mean(means <= obs))
    elif alternative == "two-sided":
        p = float(np.mean(np.abs(means) >= abs(obs)))
    else:
        raise ValueError(f"Unsupported alternative: {alternative}")
    return {"n": float(n), "mean_diff": obs, "p_value": p}


def fold_deltas(base: Mapping[str, ModelStat], other: Mapping[str, ModelStat], *, c_alternative: str,
                models: Sequence[str] = BASE_MODELS) -> pd.DataFrame:
    rows = []
    for model in models:
        pairs = {name: (base[model].scalars[key].values, other[model].scalars[key].values)
                 for name, key in DELTA_METRICS.items()}
        rows.append({"model": model, **{name: paired_signflip_test(*pair)["mean_diff"] for name, pair in pairs.items()},
                     "p_signflip_c_index": paired_signflip_test(*pairs["delta_C"], alternative=c_alternative)["p_value"]})
    return pd.DataFrame(rows)


def cohen_d_smd(high_df: pd.DataFrame, low_df: pd.DataFrame) -> pd.Series:
    n1, n0 = len(high_df), len(low_df)
    mu1, mu0 = high_df.mean(), low_df.mean()
    sd1, sd0 = high_df.std(ddof=1), low_df.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n0 - 1) * sd0**2) / (n1 + n0 - 2))
    return (mu1 - mu0) / pooled_sd.replace(0, np.nan)


def _to_index(df: pd.DataFrame, sel) -> pd.Index:
    """
    Accepts either:
      - boolean mask aligned to df.index (len == n_rows), or
      - explicit index labels / positions.
    Returns pd.Index of selected rows (labels).
    """
    if isinstance(sel, (np.ndarray, list, pd.Series)) and len(sel) == len(df.index):
        # treat as boolean mask if dtype is bool-ish
        s = np.asarray(sel)
        if s.dtype == bool:
            return df.index[s]
    return pd.Index(sel)


def _resolve_contrast_indices(df: pd.DataFrame, high_sel, low_sel, cohort_sel=None) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    Resolve cohort/high/low selections to index labels and intersect high/low with cohort.
    """
    cohort_idx = df.index if cohort_sel is None else _to_index(df, cohort_sel)
    if len(cohort_idx) < 1:
        raise ValueError("Cohort selection is empty.")
    high_idx = pd.Index(_to_index(df, high_sel)).intersection(cohort_idx)
    low_idx = pd.Index(_to_index(df, low_sel)).intersection(cohort_idx)
    if len(high_idx) < 2 or len(low_idx) < 2:
        raise ValueError(f"Not enough samples: n_high={len(high_idx)}, n_low={len(low_idx)}")
    return cohort_idx, high_idx, low_idx


def baseline_feature_value_contrast(feature_df: pd.DataFrame, high_sel, low_sel, cohort_sel=None) -> pd.Series:
    """
    Cohen's d on mean-centered baseline feature values within cohort G.
    """
    cohort_idx, high_idx, low_idx = _resolve_contrast_indices(feature_df, high_sel, low_sel, cohort_sel)
    centered = feature_df.sub(feature_df.loc[cohort_idx].mean(axis=0), axis=1)
    return cohen_d_smd(centered.loc[high_idx], centered.loc[low_idx])


def build_group_shap_share_df(shap_df: pd.DataFrame, group_map: Dict[str, str]) -> pd.DataFrame:
    """
    Compute per-line group share scores s_g(i) from SHAP values:
      s_g(i) = |sum_{j in g} phi_ij| / sum_{g'} |sum_{j in g'} phi_ij|
    """
    cols = [c for c in shap_df.columns if c in group_map]
    if not cols:
        return pd.DataFrame(index=shap_df.index)
    groups = defaultdict(list)
    for c in cols:
        groups[group_map[c]].append(c)
    group_df = pd.DataFrame({g: shap_df[gcols].sum(axis=1) for g, gcols in groups.items()}, index=shap_df.index)
    abs_group = group_df.abs()
    totals = abs_group.sum(axis=1).replace(0, np.nan)
    return abs_group.div(totals, axis=0).fillna(0.0)


def group_feature_contrast(shap_df: pd.DataFrame, group_map: Dict[str, str], high_sel, low_sel,
                           cohort_sel=None) -> pd.Series:
    """
    Cohen's d on group share scores s_g(i) between high and low risk within cohort G.
    """
    share_df = build_group_shap_share_df(shap_df, group_map)
    if share_df.empty:
        return pd.Series(dtype=float)
    _, high_idx, low_idx = _resolve_contrast_indices(share_df, high_sel, low_sel, cohort_sel)
    return cohen_d_smd(share_df.loc[high_idx], share_df.loc[low_idx])


def split_artifacts(series: pd.Series, threshold: float = 5.0) -> tuple[pd.Series, pd.Series]:
    if series.empty:
        return series, series
    artifact_mask = series.abs() > threshold
    return series[~artifact_mask], series[artifact_mask]


def split_excluded_features(series: pd.Series, excluded_features: Sequence[str]) -> tuple[pd.Series, pd.Series]:
    if series.empty:
        return series, series
    excluded = series.reindex(list(excluded_features)).dropna()
    return series.drop(index=excluded.index, errors="ignore"), excluded


def ranked(series: pd.Series, top_k: int | None = None) -> pd.Series:
    return series.reindex(series.abs().sort_values(ascending=False).index).head(len(series) if top_k is None else top_k)


def contrast_rows(cohort: str, kind: str, series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"cohort": cohort, "kind": kind, "feature": series.index, "d": series.to_numpy(dtype=float)})


def baseline_contrasts(ctx: ManuscriptContext, top_k: int = 5) -> pd.DataFrame:
    frames = []
    for cohort, cohort_mask, high, low in ctx.contrast_cohorts:
        d = baseline_feature_value_contrast(ctx.shap_x_oof, high_sel=high, low_sel=low, cohort_sel=cohort_mask)
        kept, excluded = split_excluded_features(d, MANUSCRIPT_BASELINE_CONTRAST_EXCLUDED_FEATURES)
        clean, artifacts = split_artifacts(kept)
        frames += [contrast_rows(cohort, "top", ranked(clean, top_k)), contrast_rows(cohort, "excluded", ranked(excluded)),
                   contrast_rows(cohort, "artifact", artifacts)]
    return pd.concat([frame for frame in frames if len(frame)], ignore_index=True)


def group_contrasts(ctx: ManuscriptContext, top_k: int = 5) -> pd.DataFrame:
    frames = []
    for cohort, cohort_mask, high, low in ctx.contrast_cohorts:
        d = group_feature_contrast(ctx.shap_risk_oof, group_map=ctx.group_map, high_sel=high, low_sel=low,
                                   cohort_sel=cohort_mask)
        clean, artifacts = split_artifacts(d)
        frames += [contrast_rows(cohort, "top", ranked(clean, top_k)), contrast_rows(cohort, "artifact", artifacts)]
    return pd.concat([frame for frame in frames if len(frame)], ignore_index=True)


def align_mask(mask: pd.Series | np.ndarray, target_index: pd.Index) -> np.ndarray:
    if isinstance(mask, pd.Series):
        return mask.reindex(target_index).fillna(False).to_numpy(dtype=bool)
    return np.asarray(mask, dtype=bool)


def load_feature_group_map(feature_names: Sequence[str], groups_path: str | Path = FEATURE_GROUPS_PATH) -> Dict[str, str]:
    data = json.loads(Path(groups_path).read_text())

    def matches(name: str, spec: dict, kind: str) -> bool:
        return (name in spec.get(f"{kind}_exact", []) or any(name.startswith(p) for p in spec.get(f"{kind}_prefix", []))
                or any(re.search(pattern, name) for pattern in spec.get(f"{kind}_regex", [])))

    mapping: Dict[str, str] = {}
    for group_name, spec in data.items():
        for name in map(str, feature_names):
            if matches(name, spec, "include") and not matches(name, spec, "exclude"):
                mapping[name] = group_name
    return mapping


def join_items(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + ("," if len(items) > 2 else "") + " and " + items[-1]


def feature_group_rules(path: str | Path = FEATURE_GROUPS_PATH) -> pd.DataFrame:
    rules = json.loads(Path(path).read_text())
    sentences = []
    for spec in rules.values():
        text = "; ".join(f"{one if len(spec[key]) == 1 else many} {join_items(spec[key])}"
                         for key, one, many in RULE_CLAUSES if key in spec)
        sentences.append(text[:1].upper() + text[1:] + ".")
    return pd.DataFrame({"Feature group": list(rules), "Selection rule": sentences})


def cohort_summary(dm: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, group in {"Overall": dm, **{n: dm[mask] for n, mask in subtype_masks(dm).items()}}.items():
        first = group[group["LINE"] == 1]
        rows[name] = {"Patients": group["PATIENT_ID"].nunique(), "Total mLoTs": int(group["LINE"].count()),
                      "Mean mLoTs/patient": float(group["LINE"].count() / group["PATIENT_ID"].nunique()),
                      "Event rate (%)": float(np.mean(group["PFS_EVENT"] == 1) * 100),
                      "Median PFS (days)": median_pfs(group["PFS_TIME_DAYS"], group["PFS_EVENT"]),
                      "First lines": first["PATIENT_ID"].nunique(),
                      "First-line event rate (%)": float(np.mean(first["PFS_EVENT"] == 1) * 100)}
    return pd.DataFrame.from_dict(rows, orient="index")


def baseline_characteristics(dm: pd.DataFrame) -> dict[str, float]:
    first = dm.groupby("PATIENT_ID").agg("first")
    stage_iv = first["STAGE_CDM_DERIVED_IV"]
    return {"patients": len(first), "age_mean": first["AGE"].mean(), "age_sd": first["AGE"].std(),
            "stage_iv": int(stage_iv.sum()), "stage_iv_share": stage_iv.mean()}


def age_table(dm: pd.DataFrame, patients: bool) -> pd.DataFrame:
    # Build patient-level table by dropping duplicates on patient_id (keeps first row per patient).
    d = dm.sort_values(["PATIENT_ID", "LINE"]).drop_duplicates("PATIENT_ID") if patients else dm.copy()
    d = d.copy()
    d["age_group"] = pd.cut(d.AGE, [-float("inf"), 40, 50, 65, 75, float("inf")],
                            labels=["<=40", "41-50", "51-65", "66-75", "76+"])
    rows = []
    for group, x in d.groupby("age_group", observed=True):
        rows.append({"Age group": group, "Patients" if patients else "Lines": len(x),
                     "HR positive": int(x.HR.sum()), "HER2 positive": int(x.HER2.sum()),
                     "HR positive percent": 100*x.HR.mean(), "HER2 positive percent": 100*x.HER2.mean()})
    return pd.DataFrame(rows)


def inventory() -> pd.DataFrame:
    columns = pd.read_csv(DESIGN_MATRIX_PATH, nrows=0).columns
    groups = json.loads(FEATURES_DICT_PATH.read_text())
    feature_group = {feature: group for group, features in groups.items() for feature in features}
    modalities = {group: modality for modality in MODALITIES for group in MODALITY_GROUPS[modality]}
    rows = []
    for feature in columns:
        if feature in IGNORE_COLS or any(feature.startswith(prefix) for prefix in IGNORE_PREFIXES):
            continue
        group = feature_group[feature]
        rows.append(dict(modality=modalities[group], feature=feature, feature_group=group,
                         surveillance_pattern=feature in MODALITY_GROUPS["Surveillance Patterns"]))
    table = pd.DataFrame(rows).sort_values("modality", key=lambda s: s.map(MODALITIES.index), kind="stable")
    return table.assign(modality=table.modality.str.replace("Tumor Markers", "Tumor markers")).reset_index(drop=True)


def genomic_masking(dm: pd.DataFrame) -> dict[str, int]:
    masked = dm[dm["GENOMICS_MISSING"] == 1]
    return {"masked_lines": len(masked), "lines": len(dm), "masked_patients": masked["PATIENT_ID"].nunique(),
            "patients": dm["PATIENT_ID"].nunique()}


def selected_model_configs(model_results_paths: Mapping[str, Sequence[Path]]) -> pd.DataFrame:
    rows = []
    for model in BASE_MODELS:
        counts = Counter(path.name for path in model_results_paths[model])
        rows += [{"model": model, "config": config, "count": count}
                 for config, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    return pd.DataFrame(rows)


def compute_late_start_agent_indices(dm: pd.DataFrame) -> dict[str, list[int]]:
    planned_suffixes = [col.replace("PLANNED_", "") for col in dm.columns if col.startswith("PLANNED_")]
    received_suffixes = [col.replace("RECEIVED_", "") for col in dm.columns
                         if col.startswith("RECEIVED_") and col != "RECEIVED_AGENT_IMMUNO_OTHER"]
    if set(planned_suffixes) != set(received_suffixes):
        raise ValueError("PLANNED_ and RECEIVED_ columns do not align.")
    has_late_start_agent = (dm["RECEIVED_AGENT_IMMUNO_OTHER"] == 1).to_numpy(dtype=bool)
    has_late_start_systemic_agent = has_late_start_agent.copy()
    for suffix in planned_suffixes:
        diff_mask = (dm[f"PLANNED_{suffix}"] != dm[f"RECEIVED_{suffix}"]).to_numpy(dtype=bool)
        has_late_start_agent = has_late_start_agent | diff_mask
        if not any(token in suffix for token in ("SURGERY", "RADIATION_THERAPY", "BONE TREATMENT")):
            has_late_start_systemic_agent = has_late_start_systemic_agent | diff_mask
    late_start_indices = np.where(has_late_start_agent)[0].tolist()
    late_start_systemic_indices = np.where(has_late_start_systemic_agent)[0].tolist()
    return {"late_start_indices": late_start_indices, "late_start_systemic_indices": late_start_systemic_indices,
            "late_start_local_indices": sorted(set(late_start_indices) - set(late_start_systemic_indices))}


class ManuscriptContext:
    def __init__(self, model: str = MODEL, results_root: Path = RESULTS_PATH):
        self.model = model
        self.results_root = Path(results_root)

    @cached_property
    def dm(self) -> pd.DataFrame:
        return pd.read_csv(DESIGN_MATRIX_PATH)

    @cached_property
    def no_rad_prior_df(self) -> pd.DataFrame:
        dropped_lines = pd.read_csv("data/BREAST_dropped_pfs_lines.csv")
        return dropped_lines[dropped_lines["LINE_SOURCE"] == "no_radiology_within_90_prior"].copy()

    @cached_property
    def model_perf_bundle(self) -> tuple[Dict[str, ModelStat], Dict[str, list[Path]]]:
        return build_model_perf(self.results_root, BASE_MODELS)

    @property
    def model_perf(self) -> Dict[str, ModelStat]:
        return self.model_perf_bundle[0]

    @property
    def model_results_paths(self) -> Dict[str, list[Path]]:
        return self.model_perf_bundle[1]

    @cached_property
    def oof(self) -> tuple:
        paths = self.model_results_paths[self.model]
        return load_oof_surv_and_shap(self.dm, [p / "surv_test.npz" for p in paths], [p / "shap_values.npz" for p in paths])

    @property
    def surv_oof(self) -> pd.DataFrame:
        return self.oof[0]

    @property
    def shap_oof(self) -> pd.DataFrame:
        return self.oof[1]

    @property
    def shap_x_oof(self) -> pd.DataFrame:
        return self.oof[2]

    @property
    def shap_expected_oof(self) -> pd.Series:
        return self.oof[3]

    @cached_property
    def time_grid(self) -> np.ndarray:
        return np.asarray(self.oof[4], dtype=float)

    @cached_property
    def features(self) -> list[str]:
        return list(self.oof[5])

    @cached_property
    def shap_risk_oof(self) -> pd.DataFrame:
        return -self.shap_oof

    @cached_property
    def shap_expected_risk_oof(self) -> pd.Series:
        return 1.0 - self.shap_expected_oof

    @cached_property
    def feature_labels(self) -> dict[str, str]:
        return json.loads(Path("data/feature_display_names.json").read_text())

    @cached_property
    def group_map(self) -> dict[str, str]:
        return load_feature_group_map(self.features)

    @cached_property
    def risk_index(self) -> int:
        return int(np.argmin(np.abs(self.time_grid - MANUSCRIPT_INTERMEDIATE_RISK_DAYS)))

    @property
    def risk_day(self) -> float:
        return float(self.time_grid[self.risk_index])

    @cached_property
    def risk_scores(self) -> pd.Series:
        return 1.0 - self.surv_oof.iloc[:, self.risk_index]

    @cached_property
    def risk_strata(self) -> dict[str, list[int]]:
        return tertile_strata(self.risk_scores)

    @cached_property
    def subtype_masks(self) -> dict[str, pd.Series]:
        return subtype_masks(self.dm)

    @cached_property
    def lot_masks(self) -> dict[str, pd.Series]:
        return {"mLoT1": self.dm["LINE"] == 1, "mLoT2+": self.dm["LINE"] > 1}

    @cached_property
    def late_start(self) -> dict[str, list[int]]:
        return compute_late_start_agent_indices(self.dm)

    @cached_property
    def leakage_perf_bundle(self) -> tuple[Dict[str, ModelStat], Dict[str, list[Path]]]:
        return build_model_perf(Path("leakage"), BASE_MODELS)

    @property
    def leakage_perf(self) -> Dict[str, ModelStat]:
        return self.leakage_perf_bundle[0]

    @cached_property
    def leakage_surv(self) -> pd.DataFrame:
        paths = [p / "surv_test.npz" for p in self.leakage_perf_bundle[1][self.model]]
        surv, time_grid = load_oof_surv_only(self.dm, paths)
        if not np.allclose(self.time_grid, time_grid):
            raise ValueError("Leakage OOF time grid does not match base-model time grid.")
        return surv

    @cached_property
    def leakage_deltas(self) -> pd.DataFrame:
        return fold_deltas(self.model_perf, self.leakage_perf, c_alternative="greater")

    @cached_property
    def age_ablation_perf(self) -> Dict[str, ModelStat]:
        return build_model_perf(Path("results_ablation/drop_age"), BASE_MODELS)[0]

    @cached_property
    def age_ablation_deltas(self) -> pd.DataFrame:
        return fold_deltas(self.model_perf, self.age_ablation_perf, c_alternative="less")

    @cached_property
    def modality_perf(self) -> dict[str, dict[str, dict]]:
        modality_perf: dict[str, dict[str, dict]] = {}
        for modality in MODALITY_GROUPS:
            modality_perf[modality] = {}
            for ablation_type in ("only", "exclude"):
                root = Path("results_ablation") / f"{ablation_type}_{modality.lower()}"
                models = [model for model in BASE_MODELS if (root / model / "eval_metrics.json").exists()]
                if not models:
                    continue
                model_perf, _ = build_model_perf(root, models)
                entry: dict = dict(model_perf)
                entry["best_model"] = max(model_perf.items(), key=lambda item: (
                    round(item[1].scalars[C_INDEX].mean, 3), item[1].scalars[AUC].mean))[0]
                modality_perf[modality][ablation_type] = entry
        return modality_perf

    @cached_property
    def contrast_cohorts(self) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        index = self.shap_x_oof.index
        line_values = pd.to_numeric(self.dm.loc[index, "LINE"], errors="coerce")
        cohort_defs = [(name, align_mask(self.subtype_masks[name], index)) for name in SUBTYPE_ORDER]
        cohort_defs += [("mLoT1", (line_values == 1).to_numpy(dtype=bool)), ("mLoT2+", (line_values > 1).to_numpy(dtype=bool))]
        cohorts = []
        for name, mask in cohort_defs:
            strata = tertile_strata(self.risk_scores, subset_indices=index[mask].tolist())
            cohorts.append((name, mask, index.isin(strata["high"]), index.isin(strata["low"])))
        return cohorts
