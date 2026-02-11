from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy import stats
from scipy.stats import chi2_contingency
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from scipy.stats import spearmanr
from survival_metrics import mae_pred_vs_km, rmst_from_curve

# %%
MODEL = "gbsa"
RESULTS_ROOT = Path("results")
OUTPUT_DIR = Path("figures")
from config import AUC, C_INDEX

BASE_MODELS: Sequence[str] = sorted(("coxph", "deephit", "deepsurv", "gbsa", "rsf"))


def assign_tertiles(values: pd.Series) -> pd.Series:
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


@dataclass
class OuterFoldEntry:
    name: str
    results_path: Path
    metrics: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OuterFoldEntry":
        return cls(
            name=str(data["name"]),
            results_path=Path(data["results_path"]),
            metrics=dict(data["metrics"]),
        )


@dataclass
class OuterFolds:
    folds: Dict[str, OuterFoldEntry]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "OuterFolds":
        return cls(
            folds={
                str(key): OuterFoldEntry.from_dict(value)
                for key, value in payload.items()
            }
        )

    def __getitem__(self, key: str) -> OuterFoldEntry:
        return self.folds[key]

    def keys(self):
        return self.folds.keys()

    def items(self):
        return self.folds.items()


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
    # Kaplan–Meier median PFS (accounts for censoring).
    from survival_stats import km_median_ci

    median, _, _ = km_median_ci(pfs, events, alpha=0.05)
    return median


def median_pfs_ci(
    pfs: pd.Series,
    events: pd.Series,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    from survival_stats import km_median_ci

    return km_median_ci(pfs, events, alpha=alpha)


def format_median_ci(
    median: float, lower: float, upper: float, *, digits: int = 2
) -> str:
    if np.isnan(median):
        return "NA"
    if np.isinf(median):
        return "NR"

    fmt = f"{{:.{digits}f}}"
    median_str = fmt.format(median)
    if np.isfinite(lower) and np.isfinite(upper):
        return f"{median_str} ({fmt.format(lower)}-{fmt.format(upper)})"
    return f"{median_str} (NA)"


def _pretty_combo_label(col: str, prefix: str) -> str:
    label = col
    prefix_token = f"{prefix}_"
    if label.startswith(prefix_token):
        label = label[len(prefix_token) :]
    label = re.sub(r"_+", " ", label).strip()
    return label.title()


EXCLUDED_TREATMENT_LABELS = {
    "SURGERY",
    "RADIATION_THERAPY",
    "BONE TREATMENT",
}


def _normalize_treatment_label(label: str) -> str:
    return re.sub(r"[_\s]+", " ", label).strip().upper()


def _is_excluded_treatment_label(label: str) -> bool:
    normalized = _normalize_treatment_label(label)
    padded = f" {normalized} "
    for excluded in EXCLUDED_TREATMENT_LABELS:
        if f" {excluded} " in padded:
            return True
    return False


def _aggregate_agent_treatment_categories(
    dm: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    agent_prefix = "PLANNED_AGENT_"
    agent_cols = [
        col
        for col in dm.columns
        if col.startswith(agent_prefix) and not _is_excluded_treatment_label(col)
    ]
    if not agent_cols:
        return pd.DataFrame()
    groups: dict[str, list[str]] = {}
    for col in agent_cols:
        remainder = col[len(agent_prefix) :]
        if "_" not in remainder:
            continue
        category = remainder.rsplit("_", 1)[0]
        if _is_excluded_treatment_label(category):
            continue
        groups.setdefault(category, []).append(col)
    if not groups:
        return pd.DataFrame()
    totals = {}
    for category, cols in groups.items():
        total = 0.0
        for c in cols:
            total += float(dm[c].astype(float).sum())
        totals[category] = total
    sorted_cats = sorted(totals, key=lambda key: totals[key], reverse=True)
    top_cats = sorted_cats[:top_n]
    aggregated: dict[str, pd.Series] = {}
    for category in top_cats:
        cols = groups[category]
        aggregated[category] = (
            dm[cols].fillna(0).astype(float).sum(axis=1) >= 1
        ).astype(int)
    leftover = [cat for cat in sorted_cats if cat not in top_cats]
    if leftover:
        other_cols = [col for cat in leftover for col in groups[cat]]
        aggregated["Other"] = (
            dm[other_cols].fillna(0).astype(float).sum(axis=1) >= 1
        ).astype(int)
    sanitized: dict[str, pd.Series] = {}
    for category, series in aggregated.items():
        safe = re.sub(r"[^A-Z0-9]+", "_", category.upper()).strip("_")
        if not safe:
            safe = "OTHER"
        col_name = f"PLANNED_TREATMENT_{safe}"
        sanitized[col_name] = series.astype(int)
    return pd.DataFrame(sanitized, index=dm.index)


def treatment_combinations(dm: pd.DataFrame, col_name: str, *, top_n: int = 5):
    col_key = col_name.upper()
    prefix = f"PLANNED_{col_key}"
    if col_key == "TREATMENT":
        columns_df = _aggregate_agent_treatment_categories(dm, top_n=top_n)
        columns = list(columns_df.columns)
    else:
        columns = [
            col
            for col in dm.columns
            if col.startswith(prefix) and not _is_excluded_treatment_label(col)
        ]
        columns_df = dm[columns].fillna(0).astype(int) if columns else pd.DataFrame()
    if not columns:
        print(f"No {prefix} columns available.")
        return
    columns_df = columns_df.astype(int)
    table_df = pd.concat(
        [
            dm[["HR", "HER2"]].reset_index(drop=True),
            columns_df.reset_index(drop=True),
        ],
        axis=1,
    )
    combinations = (
        table_df.groupby(["HR", "HER2"] + columns)
        .size()
        .reset_index(name="count")
        .sort_values(by=["HR", "HER2", "count"], ascending=[True, True, False])
        .assign(
            percentage=lambda df: df["count"]
            / df.groupby(["HR", "HER2"])["count"].transform("sum")
            * 100
        )
        .assign(
            TREATMENTS=lambda df: df[columns].apply(
                lambda row: ", ".join(
                    _pretty_combo_label(col, prefix)
                    for col, val in row.items()
                    if val == 1
                )
                or "None",
                axis=1,
            )
        )
        .drop(columns=columns)
    )
    for (hr, her2), group in combinations.groupby(["HR", "HER2"]):
        print(
            f"\nTop treatments for HR{'+' if hr else '-'}/HER2{'+' if her2 else '-'}:"
        )
        print(
            tabulate(
                group.head(5)[["TREATMENTS", "count", "percentage"]],
                showindex=False,
            )
        )


def treatment_combinations_to_latex_rows(
    dm: pd.DataFrame,
    col_name: str,
    top_n: int = 5,
    top_k: int = 5,
    pct_decimals: int = 1,
    indent: str = "    ",
    print_rows: bool = True,
    csv_path: str | Path | None = None,
) -> str:
    """
    Produce LaTeX table rows like:
    HR$-$/HER2$-$ & Chemo & 261 (67.1\\%) & Chemo, Other & 42 (10.8\\%) \\\\
        HR$-$/HER2$+$ & ...
    """
    col_key = col_name.upper()
    prefix = f"PLANNED_{col_key}"

    if col_key == "TREATMENT":
        columns_df = _aggregate_agent_treatment_categories(dm, top_n=top_n)
        columns = list(columns_df.columns)
    else:
        columns = [
            c
            for c in dm.columns
            if c.startswith(prefix) and not _is_excluded_treatment_label(c)
        ]
        columns_df = dm[columns].fillna(0).astype(int) if columns else pd.DataFrame()

    if not columns:
        raise ValueError(f"No {prefix} columns available.")

    columns_df = columns_df.astype(int)

    table_df = pd.concat(
        [dm[["HR", "HER2"]].reset_index(drop=True), columns_df.reset_index(drop=True)],
        axis=1,
    )

    combinations = (
        table_df.groupby(["HR", "HER2"] + columns)
        .size()
        .reset_index(name="count")
        .sort_values(by=["HR", "HER2", "count"], ascending=[True, True, False])
        .assign(
            percentage=lambda df: df["count"]
            / df.groupby(["HR", "HER2"])["count"].transform("sum")
            * 100
        )
        .assign(
            TREATMENTS=lambda df: df[columns].apply(
                lambda row: ", ".join(
                    _pretty_combo_label(col, prefix)
                    for col, val in row.items()
                    if int(val) == 1
                )
                or "None",
                axis=1,
            )
        )
        .drop(columns=columns)
    )

    if csv_path is None:
        suffix = ""
        if "LINE" in dm.columns:
            line_vals = pd.to_numeric(dm["LINE"], errors="coerce").dropna().unique()
            if line_vals.size == 1:
                line_val = line_vals[0]
                suffix = f"_line{int(line_val)}" if float(line_val).is_integer() else ""
        csv_path = (
            Path("data") / f"treatment_combinations_{col_key.lower()}{suffix}.csv"
        )
    if csv_path:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        combinations.to_csv(csv_path, index=False)

    def _latex_escape(s: str) -> str:
        # minimal LaTeX escaping for common specials
        repl = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(repl.get(ch, ch) for ch in s)

    def _hr_her2_label(hr: bool, her2: bool) -> str:
        return f"HR$\\mathrm{{{'+' if hr else '-'}}}$/HER2$\\mathrm{{{'+' if her2 else '-'}}}$"

    lines: list[str] = []
    first = True

    for (hr, her2), group in combinations.groupby(["HR", "HER2"], sort=True):
        top = group.head(top_k)

        parts: list[str] = [_hr_her2_label(bool(hr), bool(her2))]
        for _, row in top.iterrows():
            treatment = _latex_escape(str(row["TREATMENTS"]))
            count = int(row["count"])
            pct = float(row["percentage"])
            parts.append(treatment)
            parts.append(f"{count} ({pct:.{pct_decimals}f}\\%)")

        # If fewer than top_k combos exist, pad empty cells to keep column count stable
        missing = top_k - len(top)
        for _ in range(missing):
            parts.extend(["", ""])

        line = " & ".join(parts) + r" \\"
        if not first:
            line = indent + line
        first = False
        lines.append(line)

    out = "\n".join(lines)
    if print_rows:
        print(out)
    return out


# %%
def print_formatted_model_stats(
    model_perf: Mapping[str, ModelStat],
    row_order: Sequence[str] | None = None,
    mean_decimals: int = 3,
    std_decimals: int = 3,
) -> None:
    """
    Emit LaTeX rows for the manuscript table using `MODEL_PERF`-style objects.

    Each row has the first column filled only for the first metric to match the target
    snippet, with subsequent rows prefixed by '&'.
    """
    metric_rows: list[tuple[str, str, bool]] = [
        ("C-index", C_INDEX, False),
        ("mean AUC", AUC, False),
        ("AUC @ 1y", "auc_at_1y", False),
        ("AUC @ 2y", "auc_at_2y", False),
        ("IBS", "IBS", True),
    ]

    models = list(row_order) if row_order is not None else list(model_perf.keys())

    def _fmt(mean_val: float, std_val: float, bold: bool) -> str:
        inner = f"{mean_val:.{mean_decimals}f} \\pm {std_val:.{std_decimals}f}"
        return f"$\\mathbf{{{inner}}}$" if bold else f"${inner}$"

    for idx, (label, metric_key, lower_is_better) in enumerate(metric_rows):
        means: list[float] = []
        stds: list[float] = []
        for model in models:
            stat = model_perf[model].scalars.get(metric_key)
            if stat is None:
                raise KeyError(f"Missing metric {metric_key!r} for model {model!r}")
            means.append(float(stat.mean))
            stds.append(float(stat.std))

        arr = np.asarray(means, dtype=float)
        best_idx = int(np.nanargmin(arr) if lower_is_better else np.nanargmax(arr))

        cells = [
            _fmt(mean, std, bold=(j == best_idx))
            for j, (mean, std) in enumerate(zip(means, stds))
        ]

        prefix = f"{label} & " if idx == 0 else f"    & {label} & "
        print(prefix + " & ".join(cells) + r" \\")


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bootstrap_subtype_metrics(
    sub_df: pd.DataFrame,
    surv_np: np.ndarray,
    time_grid: np.ndarray,
    rmst_tau: float,
    n_boot: int,
    rng: np.random.Generator,
    *,
    time_col: str,
    event_col: str,
    patient_id_col: str,
) -> dict[str, np.ndarray]:
    pid_groups = sub_df.groupby(patient_id_col, sort=False).indices
    if not pid_groups:
        return {"mae": np.array([]), "rmst_delta": np.array([])}
    index_array = sub_df.index.to_numpy()
    pid_indices = [index_array[idxs] for idxs in pid_groups.values()]
    n_pids = len(pid_indices)

    mae_samples: list[float] = []
    rmst_delta_samples: list[float] = []
    for _ in range(n_boot):
        sampled = rng.integers(0, n_pids, size=n_pids)
        boot_indices = np.concatenate([pid_indices[i] for i in sampled])
        if boot_indices.size == 0:
            continue
        curves = surv_np[boot_indices]
        pred_mean = np.nanmean(curves, axis=0)

        rows = sub_df.loc[boot_indices]
        durations = pd.to_numeric(rows[time_col], errors="coerce").to_numpy(dtype=float)
        events = pd.to_numeric(rows[event_col], errors="coerce").to_numpy(dtype=int)
        valid = np.isfinite(durations) & np.isfinite(events)
        durations = durations[valid]
        events = events[valid]
        if durations.size == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(durations=durations, event_observed=events)
        km_series = kmf.survival_function_[kmf._label]
        km_index = km_series.index.to_numpy(dtype=float)
        km_values = km_series.to_numpy(dtype=float)
        mae_samples.append(mae_pred_vs_km(time_grid, pred_mean, km_index, km_values))
        rmst_km = float(restricted_mean_survival_time(kmf, t=rmst_tau))
        rmst_pred = rmst_from_curve(time_grid, pred_mean, rmst_tau)
        rmst_delta_samples.append(rmst_pred - rmst_km)

    return {
        "mae": np.asarray(mae_samples, dtype=float),
        "rmst_delta": np.asarray(rmst_delta_samples, dtype=float),
    }


def _summarize_bootstrap(samples: np.ndarray) -> tuple[float, float, float]:
    if samples.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.nanmean(samples))
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return mean, float(low), float(high)


def _pairwise_bootstrap_diff(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    rng: np.random.Generator,
    n_draw: int = 2000,
) -> tuple[float, float, float, float]:
    if samples_a.size == 0 or samples_b.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    n_draw = int(min(n_draw, max(samples_a.size, samples_b.size)))
    draw_a = rng.choice(samples_a, size=n_draw, replace=True)
    draw_b = rng.choice(samples_b, size=n_draw, replace=True)
    diff = draw_a - draw_b
    mean = float(np.nanmean(diff))
    low, high = np.nanpercentile(diff, [2.5, 97.5])
    p_two = 2 * min(float(np.mean(diff >= 0)), float(np.mean(diff <= 0)))
    return mean, float(low), float(high), float(p_two)


def summarize_subtype_calibration(
    subtype_outputs: Mapping[str, Mapping[str, object]],
    design_matrix: pd.DataFrame,
    surv_np: np.ndarray,
    time_grid: np.ndarray,
    rmst_tau: float,
    n_boot: int,
    random_state: int,
    subtype_order: Sequence[str],
    *,
    time_col: str = "PFS_TIME_DAYS",
    event_col: str = "PFS_EVENT",
    patient_id_col: str = "PATIENT_ID",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    surv_np = np.asarray(surv_np, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    tau = float(rmst_tau)

    subtype_rows: list[dict[str, object]] = []
    bootstrap_cache: dict[str, dict[str, np.ndarray]] = {}

    for idx, subtype in enumerate(subtype_order):
        info = subtype_outputs.get(subtype)
        if info is None:
            continue
        subtype_indices = np.asarray(info.get("indices", []), dtype=int)
        if subtype_indices.size == 0:
            continue
        sub_df = design_matrix.loc[subtype_indices]
        n_lines = int(sub_df.shape[0])
        n_patients = int(sub_df[patient_id_col].nunique())

        details = info.get("details", {})
        pop_metrics = details.get("population_metrics", {})
        mae_val = _safe_float(pop_metrics.get("mae_pred_vs_km"))

        risk_table = details.get("risk_summary_table")
        rmst_km = float("nan")
        rmst_delta = float("nan")
        if isinstance(risk_table, pd.DataFrame) and not risk_table.empty:
            pop_row = risk_table[risk_table["stratum"] == "Population"]
            if not pop_row.empty:
                row = pop_row.iloc[0]
                rmst_km = _safe_float(
                    pd.to_numeric(row.get("RMST_KM"), errors="coerce")
                )
                rmst_delta = _safe_float(
                    pd.to_numeric(row.get("RMST delta"), errors="coerce")
                )

        rng = np.random.default_rng(int(random_state) + idx + 1)
        boot_metrics = _bootstrap_subtype_metrics(
            sub_df=sub_df,
            surv_np=surv_np,
            time_grid=time_grid,
            rmst_tau=tau,
            n_boot=n_boot,
            rng=rng,
            time_col=time_col,
            event_col=event_col,
            patient_id_col=patient_id_col,
        )
        bootstrap_cache[subtype] = boot_metrics
        mae_mean, mae_low, mae_high = _summarize_bootstrap(boot_metrics["mae"])
        delta_mean, delta_low, delta_high = _summarize_bootstrap(
            boot_metrics["rmst_delta"]
        )

        subtype_rows.append(
            {
                "subtype": subtype,
                "n_lines": n_lines,
                "n_patients": n_patients,
                "mae": mae_val,
                "mae_boot_mean": mae_mean,
                "mae_ci_low": mae_low,
                "mae_ci_high": mae_high,
                "rmst_km": rmst_km,
                "rmst_delta": rmst_delta,
                "delta_boot_mean": delta_mean,
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
            }
        )

    subtype_summary_df = pd.DataFrame(subtype_rows)

    pair_rows: list[dict[str, object]] = []
    pair_rng = np.random.default_rng(int(random_state) + 101)
    for i, subtype_a in enumerate(subtype_order):
        for subtype_b in subtype_order[i + 1 :]:
            if subtype_a not in bootstrap_cache or subtype_b not in bootstrap_cache:
                continue
            mae_diff = _pairwise_bootstrap_diff(
                bootstrap_cache[subtype_a]["mae"],
                bootstrap_cache[subtype_b]["mae"],
                pair_rng,
            )
            delta_diff = _pairwise_bootstrap_diff(
                bootstrap_cache[subtype_a]["rmst_delta"],
                bootstrap_cache[subtype_b]["rmst_delta"],
                pair_rng,
            )
            pair_rows.append(
                {
                    "group_a": subtype_a,
                    "group_b": subtype_b,
                    "metric": "MAE",
                    "diff_mean": mae_diff[0],
                    "diff_ci_low": mae_diff[1],
                    "diff_ci_high": mae_diff[2],
                    "p_two_sided": mae_diff[3],
                }
            )
            pair_rows.append(
                {
                    "group_a": subtype_a,
                    "group_b": subtype_b,
                    "metric": "RMST delta",
                    "diff_mean": delta_diff[0],
                    "diff_ci_low": delta_diff[1],
                    "diff_ci_high": delta_diff[2],
                    "p_two_sided": delta_diff[3],
                }
            )
    pairwise_df = pd.DataFrame(pair_rows)

    corr_stats: dict[str, object] = {}
    if not subtype_summary_df.empty:
        corr_lines = spearmanr(subtype_summary_df["n_lines"], subtype_summary_df["mae"])
        corr_patients = spearmanr(
            subtype_summary_df["n_patients"], subtype_summary_df["mae"]
        )
        corr_stats = {
            "n_lines": {"rho": corr_lines.correlation, "p": corr_lines.pvalue},
            "n_patients": {"rho": corr_patients.correlation, "p": corr_patients.pvalue},
        }
    return subtype_summary_df, pairwise_df, corr_stats


def find_line_anthracycline_treatments(
    pfs_path: str | Path = "data/BREAST_pfs.csv",
    timeline_path: str | Path = "data/msk_chord_2024/data_timeline_treatment.tsv",
    anthracycline_agents: Sequence[str] = (
        "DOXORUBICIN",
        "EPIRUBICIN",
        "DAUNORUBICIN",
        "IDARUBICIN",
    ),
) -> pd.DataFrame:
    """Return anthracycline treatment events overlapping each PFS line window."""
    pfs_df = pd.read_csv(pfs_path)
    tx_df = pd.read_csv(timeline_path, sep="\t")

    required_pfs = {"PATIENT_ID", "LINE", "LINE_START", "EVENT_DAY"}
    required_tx = {"PATIENT_ID", "START_DATE", "STOP_DATE", "AGENT"}
    missing_pfs = required_pfs - set(pfs_df.columns)
    missing_tx = required_tx - set(tx_df.columns)
    if missing_pfs:
        raise ValueError(
            f"Missing required columns in {pfs_path}: {sorted(missing_pfs)}"
        )
    if missing_tx:
        raise ValueError(
            f"Missing required columns in {timeline_path}: {sorted(missing_tx)}"
        )

    tx_work = tx_df.copy()
    if "EVENT_TYPE" in tx_work.columns:
        tx_work = tx_work[
            tx_work["EVENT_TYPE"].astype(str).str.upper().eq("TREATMENT")
        ].copy()

    agent_regex = "|".join(re.escape(agent) for agent in anthracycline_agents)
    tx_work["IS_ANTHRACYCLINE"] = (
        tx_work["AGENT"]
        .astype(str)
        .str.contains(
            agent_regex,
            case=False,
            na=False,
            regex=True,
        )
    )
    tx_work = tx_work[tx_work["IS_ANTHRACYCLINE"]].copy()

    merged = pfs_df.merge(tx_work, on="PATIENT_ID", how="inner")
    window_start = merged[["LINE_START", "EVENT_DAY"]].min(axis=1)
    window_end = merged[["LINE_START", "EVENT_DAY"]].max(axis=1)
    tx_start = merged["START_DATE"]
    tx_end = merged["STOP_DATE"].fillna(merged["START_DATE"])

    # Keep treatment intervals that intersect the per-line [LINE_START, EVENT_DAY] window.
    overlap_mask = (tx_start <= window_end) & (tx_end >= window_start)
    matched = merged.loc[overlap_mask].copy()
    matched["OVERLAP_START"] = np.maximum(
        tx_start[overlap_mask], window_start[overlap_mask]
    )
    matched["OVERLAP_END"] = np.minimum(tx_end[overlap_mask], window_end[overlap_mask])
    matched["OVERLAP_DAYS"] = (
        matched["OVERLAP_END"] - matched["OVERLAP_START"] + 1
    ).clip(lower=0)

    cols = [
        "PATIENT_ID",
        "LINE",
        "LINE_START",
        "EVENT_DAY",
        "START_DATE",
        "STOP_DATE",
        "OVERLAP_START",
        "OVERLAP_END",
        "OVERLAP_DAYS",
        "AGENT",
    ]
    keep_cols = [c for c in cols if c in matched.columns]
    return (
        matched[keep_cols]
        .sort_values(["PATIENT_ID", "LINE", "OVERLAP_START", "AGENT"])
        .reset_index(drop=True)
    )


def median_lines_within_window_from_anchor_line(
    design_matrix_path: str | Path = "data/design_matrix.csv",
    anchor_line: int = 1,
    window_days: int = 365,
) -> dict[str, Any]:
    """Compute per-patient line counts within `window_days` after an anchor line start."""
    dm = pd.read_csv(design_matrix_path)
    required_cols = {"PATIENT_ID", "LINE", "LINE_START"}
    missing_cols = required_cols - set(dm.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {design_matrix_path}: {sorted(missing_cols)}"
        )

    work = dm.loc[:, ["PATIENT_ID", "LINE", "LINE_START"]].copy()
    work["LINE"] = pd.to_numeric(work["LINE"], errors="coerce")
    work["LINE_START"] = pd.to_numeric(work["LINE_START"], errors="coerce")
    work = work.dropna(subset=["PATIENT_ID", "LINE", "LINE_START"]).copy()
    if work.empty:
        raise ValueError("No valid line rows found in design_matrix.csv.")
    work["LINE"] = work["LINE"].astype(int)

    anchor = (
        work.loc[work["LINE"] == int(anchor_line), ["PATIENT_ID", "LINE_START"]]
        .drop_duplicates(subset=["PATIENT_ID"])
        .rename(columns={"LINE_START": "ANCHOR_LINE_START"})
    )
    if anchor.empty:
        raise ValueError(
            f"No modeled LoT{int(anchor_line)} rows found in design_matrix.csv."
        )

    candidate_lines = work.loc[
        work["PATIENT_ID"].isin(anchor["PATIENT_ID"]),
        ["PATIENT_ID", "LINE", "LINE_START"],
    ].copy()
    candidate_lines = candidate_lines.merge(anchor, on="PATIENT_ID", how="inner")

    line_start = candidate_lines["LINE_START"].to_numpy(dtype=float)
    anchor_start = candidate_lines["ANCHOR_LINE_START"].to_numpy(dtype=float)
    in_window = (line_start >= anchor_start) & (
        line_start <= anchor_start + float(window_days)
    )
    window_df = candidate_lines.loc[in_window].copy()

    counts = (
        window_df.groupby("PATIENT_ID")
        .size()
        .rename("LINES_WITHIN_WINDOW")
        .reset_index()
    )
    max_line = (
        window_df.groupby("PATIENT_ID")["LINE"]
        .max()
        .rename("MAX_LINE_WITHIN_WINDOW")
        .reset_index()
    )
    counts = (
        anchor[["PATIENT_ID"]]
        .merge(counts, on="PATIENT_ID", how="left")
        .merge(max_line, on="PATIENT_ID", how="left")
    )
    counts["LINES_WITHIN_WINDOW"] = counts["LINES_WITHIN_WINDOW"].fillna(0).astype(int)
    counts["MAX_LINE_WITHIN_WINDOW"] = (
        counts["MAX_LINE_WITHIN_WINDOW"].fillna(0).astype(int)
    )

    return {
        "anchor_line": int(anchor_line),
        "window_days": int(window_days),
        "n_patients": int(counts["PATIENT_ID"].nunique()),
        "median_lines_within_window": float(counts["LINES_WITHIN_WINDOW"].median()),
        "median_max_line_within_window": float(
            counts["MAX_LINE_WITHIN_WINDOW"].median()
        ),
        "per_patient_counts": counts.sort_values("PATIENT_ID").reset_index(drop=True),
        "counts": counts,
    }


def median_lines_within_first_year_from_mlot1(
    design_matrix_path: str | Path = "data/design_matrix.csv",
    window_days: int = 365,
) -> dict[str, Any]:
    return median_lines_within_window_from_anchor_line(
        design_matrix_path=design_matrix_path,
        anchor_line=1,
        window_days=window_days,
    )


def median_lines_within_first_year_from_mlot2(
    design_matrix_path: str | Path = "data/design_matrix.csv",
    window_days: int = 365,
) -> dict[str, Any]:
    return median_lines_within_window_from_anchor_line(
        design_matrix_path=design_matrix_path,
        anchor_line=2,
        window_days=window_days,
    )


def _build_time_series_from_folds(
    folds: OuterFolds, metric_key: str
) -> TimeMetricSeries | None:
    arrays: list[np.ndarray] = []
    times = None
    for _, fold in folds.items():
        fm = fold.metrics
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
    return TimeMetricSeries(
        times=times,
        mean=stacked.mean(axis=0),
        std=stacked.std(axis=0),
        n=stacked.shape[0],
    )


def _build_hrher2_series(folds: OuterFolds) -> Dict[str, Dict[str, TimeMetricSeries]]:
    accum: dict[str, dict[str, list[np.ndarray]]] = {"auc": {}, "ipcw": {}}
    sample_times: dict[str, dict[str, np.ndarray]] = {"auc": {}, "ipcw": {}}

    for _, fold in folds.items():
        hrher_block = fold.metrics.get("hr_her2_metrics", {})
        for subgroup, stats in hrher_block.items():
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
            sub_map[subgroup] = TimeMetricSeries(
                times=sample_times[metric_key][subgroup],
                mean=stacked.mean(axis=0),
                std=stacked.std(axis=0),
                n=stacked.shape[0],
            )
        if sub_map:
            hrher_map[metric_key] = sub_map
    return hrher_map


def build_model_perf(
    results_root: Path, base_models: Sequence[str]
) -> tuple[Dict[str, ModelStat], Dict[str, list[Path]]]:
    model_perf: Dict[str, ModelStat] = {}
    model_results_paths: Dict[str, list[Path]] = {}
    for model in base_models:
        folds = OuterFolds.from_json(
            json.load(open(results_root / model / "eval_metrics.json"))["outer_folds"]
        )
        model_results_paths[model] = [fold.results_path for _, fold in folds.items()]
        scalars: Dict[str, ScalarMetricSummary] = {}

        def _add_scalar(name: str, values: list[float]):
            scalars[name] = ScalarMetricSummary(
                mean=float(np.mean(values)),
                std=float(np.std(values)),
                values=[float(v) for v in values],
            )

        vals_c = []
        vals_auc = []
        vals_ibs = []
        vals_p = []
        auc90 = []
        auc180 = []
        auc1y = []
        auc2y = []
        for _, fold in folds.items():
            fm = fold.metrics
            vals_c.append(fm[C_INDEX])
            vals_auc.append(fm[AUC])
            vals_ibs.append(fm["IBS"])
            vals_p.append(fm["p_value"])
            horizons = fm["eval_horizons"]
            auc_arr = fm["auc"]
            auc90.append(
                float(np.mean([auc_arr[i] for i, x in enumerate(horizons) if x == 90]))
            )
            auc180.append(
                float(np.mean([auc_arr[i] for i, x in enumerate(horizons) if x == 180]))
            )
            horizons_arr = np.asarray(horizons, dtype=float)
            auc1y_idx = int(np.argmin(np.abs(horizons_arr - 365.0)))
            auc1y.append(float(auc_arr[auc1y_idx]))
            auc2y.append(float(auc_arr[-1]))

        _add_scalar(C_INDEX, vals_c)
        _add_scalar(AUC, vals_auc)
        _add_scalar("IBS", vals_ibs)
        _add_scalar("p_value", vals_p)
        _add_scalar("auc_at_90d", auc90)
        _add_scalar("auc_at_180d", auc180)
        _add_scalar("auc_at_1y", auc1y)
        _add_scalar("auc_at_2y", auc2y)

        series: Dict[str, TimeMetricSeries] = {}
        for metric_key in ("auc", "ipcw"):
            ts = _build_time_series_from_folds(folds, metric_key)
            if ts:
                series[metric_key] = ts

        hrher_series = _build_hrher2_series(folds)
        model_perf[model] = ModelStat(
            name=model,
            scalars=scalars,
            series=series,
            hrher2_series=hrher_series,
        )
    return model_perf, model_results_paths


def load_oof_surv_only(dm, surv_paths):
    # --- build common time grid ---
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

    # --- initialize arrays ---
    n = len(dm)
    surv_oof = np.full((n, common_time.size), np.nan, dtype=float)

    # --- fill OOF arrays ---
    for surv_path in surv_paths:
        with np.load(surv_path) as sf:
            idx_test = np.asarray(sf["idx_test"], dtype=int)
            time_grid = np.asarray(sf["time"], dtype=float)
            surv_vals = np.asarray(sf["surv"], dtype=float)
            # align columns to the common grid
            col_idx = np.searchsorted(time_grid, common_time)
            if not np.allclose(time_grid[col_idx], common_time):
                raise ValueError(f"Time grid mismatch in {surv_path}")
            surv_oof[idx_test] = surv_vals[:, col_idx]

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
            shap_vals = np.asarray(hf["shap_values"], dtype=float)
            shap_X_vals = np.asarray(hf["X"], dtype=float)
            expected_vals = np.asarray(hf["expected_value"]).reshape(-1)
            shap_oof[sample_idx] = shap_vals
            shap_X[sample_idx] = shap_X_vals
            shap_expected[sample_idx] = expected_vals

    # --- wrap as DataFrames/Series for downstream use ---
    shap_df = pd.DataFrame(shap_oof, index=dm.index, columns=feature_names)
    shap_X_df = pd.DataFrame(shap_X, index=dm.index, columns=feature_names)
    shap_expected_s = pd.Series(shap_expected, index=dm.index, name="expected_value")

    return surv_df, shap_df, shap_X_df, shap_expected_s, common_time, feature_names


def paired_signflip_test(
    x: Sequence[float],
    y: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> Dict[str, float]:
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
    flips = np.asarray(
        [((mask >> np.arange(n)) & 1) * 2 - 1 for mask in range(1 << n)],
        dtype=float,
    )
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


from scipy import stats


def _metric_values(
    model_perf: Mapping[str, Any], model: str, metric: str
) -> np.ndarray:
    stat = model_perf[model].scalars.get(metric)
    if stat is None:
        raise KeyError(f"Missing metric {metric!r} for model {model!r}")
    return np.asarray(stat.values, dtype=float)


def paired_tests_summary(
    clean: Mapping[str, Any],
    leaky: Mapping[str, Any],
    models: Sequence[str],
    metrics: Sequence[str],
    alternative: str = "greater",
    metric_alternatives: Dict[str, str] | None = None,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for model in models:
        for metric in metrics:
            x = _metric_values(clean, model, metric)
            y = _metric_values(leaky, model, metric)
            alt = (
                metric_alternatives.get(metric, alternative)
                if metric_alternatives is not None
                else alternative
            )
            signflip = paired_signflip_test(x, y, alternative=alt)
            row: Dict[str, Any] = {
                "model": model,
                "metric": metric,
                "alternative": alt,
                "n_folds": int(signflip["n"]),
                "clean_mean": float(np.nanmean(x)),
                "leaky_mean": float(np.nanmean(y)),
                "delta_leaky_minus_clean": float(signflip["mean_diff"]),
                "p_signflip": float(signflip["p_value"]),
            }
            if stats is not None:
                try:
                    try:
                        row["p_ttest_rel"] = float(
                            stats.ttest_rel(
                                y, x, nan_policy="omit", alternative=alt
                            ).pvalue
                        )
                    except TypeError:
                        row["p_ttest_rel"] = float(
                            stats.ttest_rel(y, x, nan_policy="omit").pvalue
                        )
                except Exception:
                    row["p_ttest_rel"] = float("nan")
                try:
                    try:
                        row["p_wilcoxon"] = float(
                            stats.wilcoxon(
                                y,
                                x,
                                zero_method="wilcox",
                                alternative=alt,
                            ).pvalue
                        )
                    except TypeError:
                        row["p_wilcoxon"] = float(
                            stats.wilcoxon(y, x, zero_method="wilcox").pvalue
                        )
                except Exception:
                    row["p_wilcoxon"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["metric", "model"]).reset_index(drop=True)


def delta_table_from_paired(
    paired: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    subset = paired[paired["metric"].isin(metrics)].copy()
    if subset.empty:
        return pd.DataFrame()
    pivot = subset.pivot(
        index="model", columns="metric", values="delta_leaky_minus_clean"
    )
    pivot = pivot.rename(
        columns={
            C_INDEX: "delta_C",
            AUC: "delta_AUC",
            "IBS": "delta_IBS",
        }
    )
    c_index_p = subset[subset["metric"] == C_INDEX].set_index("model")["p_signflip"]
    pivot["p_signflip_c_index"] = c_index_p
    return pivot.reset_index()


def _resolve_time_index(times: np.ndarray, horizon: float) -> int:
    """Pick the index matching `horizon` if present; otherwise nearest index."""
    times = np.asarray(times, dtype=float)
    exact = np.where(times == float(horizon))[0]
    if exact.size:
        return int(exact[0])
    return int(np.argmin(np.abs(times - float(horizon))))


DEFAULT_MODEL_NAME_MAP = {
    "coxph": "CoxPH",
    "deephit": "DeepHit",
    "deepsurv": "DeepSurv",
    "gbsa": "GBSA",
    "rsf": "RSF",
}


def print_time_dependent_metric_rows(
    model_perf: Mapping[str, "ModelStat"],
    metric_key: str,
    horizons: Sequence[float] = (90, 180, 365, 730),
    decimals: int = 3,
) -> None:
    """
    Print LaTeX table rows:
    Model & mean ± std @h1 & mean ± std @h2 & ... \\\\
    Uses model_perf[model].series[metric_key].mean/std at the requested horizons.
    """
    name_map = dict(DEFAULT_MODEL_NAME_MAP)
    models = list(model_perf.keys())

    for m in models:
        if m not in model_perf or metric_key not in model_perf[m].series:
            raise KeyError(f"Missing series[{metric_key!r}] for model {m!r}")

        ts = model_perf[m].series[metric_key]
        cells = []
        for h in horizons:
            idx = _resolve_time_index(ts.times, h)
            mean = float(ts.mean[idx])
            std = float(ts.std[idx])
            cells.append(f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}")

        display_name = name_map.get(m, m.upper())
        print(f"{display_name:<8} & " + " & ".join(cells) + r" \\")


def print_time_dependent_auc_and_c(
    model_perf: Mapping[str, "ModelStat"],
    horizons: Sequence[float] = (90, 180, 365, 730),
    decimals: int = 3,
) -> None:
    """Print two blocks: IPCW-C then AUC."""
    print_time_dependent_metric_rows(
        model_perf,
        "ipcw",
        horizons=horizons,
        decimals=decimals,
    )
    print()
    print_time_dependent_metric_rows(
        model_perf,
        "auc",
        horizons=horizons,
        decimals=decimals,
    )


# def _add_scalar(name: str, values: list[float]):
#     scalars[name] = ScalarMetricSummary(
#         mean=float(np.mean(values)),
#         std=float(np.std(values)),
#         values=[float(v) for v in values],
#     )


PATTERNS = {
    # Imaging OHE (+ missing)
    "CANCER_{region}_IMAGING_STATUS": re.compile(
        r"^CANCER_(CHEST|ABDOMEN|PELVIS|HEAD|OTHER)_(IMAGED_(Y|N|INDET)_STATUS|MISSING)$"
    ),
    # Tri-states
    "PDL1_STATUS": re.compile(r"^PDL1_(POS|NEG|UNKNOWN)$"),
    "MMR_STATUS": re.compile(r"^MMR_(POS|NEG|UNKNOWN)$"),
    # Diagnosis OHE blocks
    "CLINICAL_GROUP": re.compile(r"^CLINICAL_GROUP_"),
    "PATH_GROUP": re.compile(r"^PATH_GROUP_"),
    "SUMMARY_STAGE": re.compile(r"^SUMMARY_"),
    "HISTOLOGY": re.compile(r"^HISTOLOGIC_"),
    "CANCER_SITE_SUBSITE": re.compile(r"^CANCER_SITE_SUBSITE_"),
}


def build_ohe_groups(columns):
    groups = defaultdict(list)
    for c in columns:
        m = PATTERNS["CANCER_{region}_IMAGING_STATUS"].match(c)
        if m:
            region = m.group(1)
            groups[f"CANCER_{region}_IMAGING_STATUS"].append(c)
            continue
        for name, pat in PATTERNS.items():
            if name == "CANCER_{region}_IMAGING_STATUS":
                continue
            if pat.search(c):
                groups[name].append(c)
                break
    return dict(groups)


def aggregate_ohe_arrays(
    shap_values: pd.DataFrame,
    X: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple["pd.DataFrame | np.ndarray", "pd.DataFrame | np.ndarray", list[str]]:
    """
    Aggregate OHE blocks by summing SHAP and X for features in the same group.
    - If inputs are DataFrames and return_dataframes=True, returns DataFrames with same index.
    - feature_names defaults to DataFrame columns when available.
    """

    from plots import _build_imaging_ohe_groups, _load_features_dict

    # --- Normalize inputs to numpy + capture index/columns ---
    index = shap_values.index
    shap_mat = shap_values.to_numpy(dtype=float)
    X_mat = X.loc[:, feature_names].to_numpy(dtype=float)

    # --- Validate shapes ---
    if shap_mat.ndim == 1:
        shap_mat = shap_mat.reshape(1, -1)
    if X_mat.ndim == 1:
        X_mat = X_mat.reshape(1, -1)
    if shap_mat.shape != X_mat.shape:
        raise ValueError("shap_values and X must have identical shapes.")
    if shap_mat.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match column count.")

    # --- Build group definitions ---
    features_dict = _load_features_dict(Path("data/features_dict.json"))
    feature_set = set(feature_names)
    group_definitions: dict[str, list[str]] = {}

    def register_group(group_name: str, members: Sequence[str]) -> None:
        kept = [m for m in members if m in feature_set]
        if len(kept) >= 2:
            group_definitions[group_name] = kept

    # Built-in groups from features_dict.json
    for block in ("PDL1", "MMR", "DIAGNOSIS"):
        register_group(block, features_dict.get(block, []))

    # Imaging groups
    for group_name, members in _build_imaging_ohe_groups(list(feature_names)).items():
        register_group(group_name, members)

    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    member_to_group: dict[str, str] = {}
    for group_name, members in group_definitions.items():
        for member in members:
            member_to_group[member] = group_name

    # --- Aggregate ---
    agg_shap = []
    agg_X = []
    agg_names = []
    seen_groups: set[str] = set()

    for name in feature_names:
        group = member_to_group.get(name)
        if group:
            if group in seen_groups:
                continue
            idxs = [name_to_idx[m] for m in group_definitions[group]]
            agg_shap.append(shap_mat[:, idxs].sum(axis=1))
            agg_X.append(X_mat[:, idxs].sum(axis=1))
            agg_names.append(group)
            seen_groups.add(group)
        else:
            col = name_to_idx[name]
            agg_shap.append(shap_mat[:, col])
            agg_X.append(X_mat[:, col])
            agg_names.append(name)

    shap_agg = np.column_stack(agg_shap)
    X_agg = np.column_stack(agg_X)

    shap_out = pd.DataFrame(shap_agg, index=index, columns=agg_names)
    X_out = pd.DataFrame(X_agg, index=index, columns=agg_names)
    return shap_out, X_out, agg_names


def latex_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def cohen_d_smd(high_df: pd.DataFrame, low_df: pd.DataFrame) -> pd.Series:
    n1, n0 = len(high_df), len(low_df)
    mu1, mu0 = high_df.mean(), low_df.mean()
    sd1, sd0 = high_df.std(ddof=1), low_df.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n0 - 1) * sd0**2) / (n1 + n0 - 2))
    return (mu1 - mu0) / pooled_sd.replace(0, np.nan)




def _to_index(df: pd.DataFrame, sel: IndexLike) -> pd.Index:
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


def _resolve_contrast_indices(
    df: pd.DataFrame,
    high_sel: IndexLike,
    low_sel: IndexLike,
    cohort_sel: IndexLike | None = None,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """
    Resolve cohort/high/low selections to index labels and intersect high/low with cohort.
    """
    cohort_idx = df.index if cohort_sel is None else _to_index(df, cohort_sel)
    if len(cohort_idx) < 1:
        raise ValueError("Cohort selection is empty.")

    high_idx = _to_index(df, high_sel)
    low_idx = _to_index(df, low_sel)
    high_idx = pd.Index(high_idx).intersection(cohort_idx)
    low_idx = pd.Index(low_idx).intersection(cohort_idx)
    if len(high_idx) < 2 or len(low_idx) < 2:
        raise ValueError(
            f"Not enough samples: n_high={len(high_idx)}, n_low={len(low_idx)}"
        )
    return cohort_idx, high_idx, low_idx


def baseline_feature_value_contrast(
    feature_df: pd.DataFrame,
    high_sel: IndexLike,
    low_sel: IndexLike,
    cohort_sel: IndexLike | None = None,
) -> pd.Series:
    """
    Cohen's d on mean-centered baseline feature values within cohort G.
    """
    cohort_idx, high_idx, low_idx = _resolve_contrast_indices(
        feature_df, high_sel, low_sel, cohort_sel
    )
    centered = feature_df.sub(feature_df.loc[cohort_idx].mean(axis=0), axis=1)
    return cohen_d_smd(centered.loc[high_idx], centered.loc[low_idx])


def baseline_feature_shap_contrast(
    shap_df: pd.DataFrame,
    high_sel: IndexLike,
    low_sel: IndexLike,
    cohort_sel: IndexLike | None = None,
) -> pd.Series:
    """
    Cohen's d on absolute SHAP magnitudes within cohort G (no re-centering).
    """
    _, high_idx, low_idx = _resolve_contrast_indices(
        shap_df, high_sel, low_sel, cohort_sel
    )
    abs_shap = shap_df.abs()
    return cohen_d_smd(abs_shap.loc[high_idx], abs_shap.loc[low_idx])


def build_group_shap_share_df(
    shap_df: pd.DataFrame,
    group_map: Dict[str, str],
) -> pd.DataFrame:
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

    group_sums = {g: shap_df[gcols].sum(axis=1) for g, gcols in groups.items()}
    group_df = pd.DataFrame(group_sums, index=shap_df.index)
    abs_group = group_df.abs()
    totals = abs_group.sum(axis=1).replace(0, np.nan)
    return abs_group.div(totals, axis=0).fillna(0.0)


def group_feature_contrast(
    shap_df: pd.DataFrame,
    group_map: Dict[str, str],
    high_sel: IndexLike,
    low_sel: IndexLike,
    cohort_sel: IndexLike | None = None,
) -> pd.Series:
    """
    Cohen's d on group share scores s_g(i) between high and low risk within cohort G.
    """
    share_df = build_group_shap_share_df(shap_df, group_map)
    if share_df.empty:
        return pd.Series(dtype=float)
    _, high_idx, low_idx = _resolve_contrast_indices(
        share_df, high_sel, low_sel, cohort_sel
    )
    return cohen_d_smd(share_df.loc[high_idx], share_df.loc[low_idx])




def pfs_event_prevalence_by_agebin(
    dm: pd.DataFrame,
    age_col: str = "AGE",
    event_col: str = "PFS_EVENT",
    bin_width: int = 5,
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - Age bin (Interval string like "(20, 25]")
      - Event (%) (float)
      - n (int)
    """
    df = dm.dropna(subset=[age_col, event_col]).copy()
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna(subset=[age_col, event_col])

    if df.empty:
        return pd.DataFrame(columns=["Age bin", "Event (%)", "n"])

    min_age = float(df[age_col].min())
    max_age = float(df[age_col].max())

    start = int(np.floor(min_age / bin_width) * bin_width)
    end = int(np.ceil(max_age / bin_width) * bin_width)
    edges = list(range(start, end + bin_width, bin_width))

    df["age_bin"] = pd.cut(df[age_col], bins=edges, right=True, include_lowest=False)

    out = (
        df.groupby("age_bin", observed=True)[event_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "Event (%)", "count": "n"})
        .reset_index()
    )
    out["Event (%)"] = out["Event (%)"] * 100.0
    out["Age bin"] = out["age_bin"].astype(str)
    out = out.drop(columns=["age_bin"])
    return out[["Age bin", "Event (%)", "n"]]


def print_pfs_event_prevalence_latex(
    dm: pd.DataFrame,
    age_col: str = "AGE",
    event_col: str = "PFS_EVENT",
    bin_width: int = 5,
    decimals: int = 2,
) -> None:
    """Print lines: (20, 25] & 50.00 \\\\"""
    tbl = pfs_event_prevalence_by_agebin(
        dm, age_col=age_col, event_col=event_col, bin_width=bin_width
    )
    for _, row in tbl.iterrows():
        print(f"{row['Age bin']} & {float(row['Event (%)']):.{decimals}f} \\\\")


def _to_bool(series: pd.Series) -> pd.Series:
    """Best-effort coercion to boolean for HR/HER2."""
    if pd.api.types.is_bool_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) > 0
    # object/string
    s = series.astype(str).str.strip().str.lower()
    pos = {"1", "true", "t", "yes", "y", "pos", "positive", "+", "hr+", "her2+"}
    neg = {"0", "false", "f", "no", "n", "neg", "negative", "-", "hr-", "her2-"}
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out[s.isin(pos)] = True
    out[s.isin(neg)] = False
    # fallback: try numeric parsing
    num = pd.to_numeric(series, errors="coerce")
    out = out.fillna(num.fillna(0).astype(float) > 0)
    return out.astype(bool)


def print_hr_her2_by_age_midrule_block(
    dm: pd.DataFrame,
    age_col: str = "AGE",
    hr_col: str = "HR",
    her2_col: str = "HER2",
    patient_id_col: str = "PATIENT_ID",
    decimals: int = 1,
) -> None:
    """
    Prints ONLY the LaTeX rows between \\midrule and \\bottomrule, plus chi-square outputs (unformatted).
    """
    df = dm[[age_col, hr_col, her2_col, patient_id_col]].copy()
    AGE_BINS = [0, 40, 50, 65, 75, 200]
    AGE_LABELS = [r"$\leq$40", "41--50", "51--65", "66--75", "76+"]

    df["AGE_BIN"] = pd.cut(
        df[age_col],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
        right=True,
    )

    df[hr_col] = _to_bool(df[hr_col])
    df[her2_col] = _to_bool(df[her2_col])

    # unique patients per (age_bin, hr) etc.
    # (if PATIENT_ID is already unique per row, this still works)
    by_age_hr = (
        df.dropna(subset=["AGE_BIN"])
        .groupby(["AGE_BIN", hr_col])[patient_id_col]
        .nunique()
        .unstack(hr_col, fill_value=0)
    )
    by_age_her2 = (
        df.dropna(subset=["AGE_BIN"])
        .groupby(["AGE_BIN", her2_col])[patient_id_col]
        .nunique()
        .unstack(her2_col, fill_value=0)
    )

    # Ensure both columns exist (False/True)
    for tab in (by_age_hr, by_age_her2):
        if False not in tab.columns:
            tab[False] = 0
        if True not in tab.columns:
            tab[True] = 0
        tab.sort_index(axis=1, inplace=True)

    # Percent within age bin
    hr_total = by_age_hr[False] + by_age_hr[True]
    her2_total = by_age_her2[False] + by_age_her2[True]

    hr_pos = (by_age_hr[True] / hr_total.replace(0, np.nan) * 100).fillna(0.0)
    hr_neg = 100.0 - hr_pos
    her2_pos = (by_age_her2[True] / her2_total.replace(0, np.nan) * 100).fillna(0.0)
    her2_neg = 100.0 - her2_pos

    # Print LaTeX rows
    for label in AGE_LABELS:
        print(
            f"{label} & "
            f"{hr_pos.loc[label]:.{decimals}f} & {hr_neg.loc[label]:.{decimals}f} & "
            f"{her2_pos.loc[label]:.{decimals}f} & {her2_neg.loc[label]:.{decimals}f} \\\\"
        )

    # Chi-square tests (age group vs HR; age group vs HER2), using unique patients
    # Build patient-level table by dropping duplicates on patient_id (keeps first row per patient).
    pat = df.dropna(subset=["AGE_BIN"]).drop_duplicates(subset=[patient_id_col])

    hr_ct = pd.crosstab(pat["AGE_BIN"], pat[hr_col])
    her2_ct = pd.crosstab(pat["AGE_BIN"], pat[her2_col])

    chi2_hr, p_hr, dof_hr, _ = chi2_contingency(hr_ct.values)
    chi2_her2, p_her2, dof_her2, _ = chi2_contingency(her2_ct.values)

    print("\nChi-square tests (raw):")
    print(f"HR vs age group: chi2={chi2_hr}, dof={dof_hr}, p={p_hr}")
    print(f"HER2 vs age group: chi2={chi2_her2}, dof={dof_her2}, p={p_her2}")
