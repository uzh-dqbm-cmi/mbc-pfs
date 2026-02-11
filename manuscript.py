# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Sequence
import numpy as np
import pandas as pd
from IPython.display import Image, display
from tabulate import tabulate

from config import ADMIN_CENSOR_DAYS
from build_ablation_configs import write_ablation_configs
from plots import (
    EVENT_COL,
    SUBTYPE_ORDER,
    TIME_COL,
    plot_full_group_individual_time_curves,
    plot_group_mean_abs_shap,
    plot_hrher2_time_curves,
    plot_km_late_start_vs_rest,
    plot_km_modeled_vs_no_rad_prior,
    plot_mean_surv_late_start,
    plot_mean_surv_late_start_systemic_vs_local_clean_leakage,
    plot_population_km_vs_mean,
    plot_shap_summary_from_arrays,
    plot_single_sample_shap_waterfall_from_arrays,
)
from manuscript_utils import *

# %%
MODEL = "gbsa"
RESULTS_ROOT = Path("results")
OUTPUT_DIR = Path("figures")
from config import AUC, C_INDEX

BASE_MODELS: Sequence[str] = sorted(("coxph", "deephit", "deepsurv", "gbsa", "rsf"))
LEAKAGE_MODELS: Sequence[str] = BASE_MODELS.copy()
ABLATION_MODELS: Sequence[str] = BASE_MODELS.copy()
DEFAULT_MODEL_NAME_MAP = {
    "coxph": "CoxPH",
    "deephit": "DeepHit",
    "deepsurv": "DeepSurv",
    "gbsa": "GBSA",
    "rsf": "RSF",
}


def _print_section(title: str) -> None:
    rule = "-" * 80
    print(f"\n{rule}\n{title}\n{rule}")


dm = pd.read_csv("data/design_matrix.csv")

GLOBAL_PATIENTS = dm["PATIENT_ID"].nunique()
GLOBAL_LINES = dm["LINE"].count()
GLOBAL_MEAN_LINES = GLOBAL_LINES / GLOBAL_PATIENTS
GLOBAL_EVENT_RATE = float(np.mean(dm["PFS_EVENT"] == 1) * 100)
GLOBAL_MEDIAN, GLOBAL_MEDIAN_LOW, GLOBAL_MEDIAN_HIGH = median_pfs_ci(
    dm["PFS_TIME_DAYS"], dm["PFS_EVENT"]
)
GLOBAL_MEDIAN_PFS = format_median_ci(
    GLOBAL_MEDIAN, GLOBAL_MEDIAN_LOW, GLOBAL_MEDIAN_HIGH
)

_dm_l1 = dm[dm["LINE"] == 1]
GLOBAL_L1_PATIENTS = _dm_l1["PATIENT_ID"].nunique()
GLOBAL_L1_EVENT_RATE = float(np.mean(_dm_l1["PFS_EVENT"] == 1) * 100)
GLOBAL_L1_MEDIAN, GLOBAL_L1_MEDIAN_LOW, GLOBAL_L1_MEDIAN_HIGH = median_pfs_ci(
    _dm_l1["PFS_TIME_DAYS"], _dm_l1["PFS_EVENT"]
)
GLOBAL_L1_MEDIAN_PFS = format_median_ci(
    GLOBAL_L1_MEDIAN, GLOBAL_L1_MEDIAN_LOW, GLOBAL_L1_MEDIAN_HIGH
)
print("Overall cohort statistics:")
print(
    "Used for `sn-article.tex`: Table `table:cohort-summary` (Global row + subtype rows)."
)
print(
    f"Patients: {GLOBAL_PATIENTS}, Lines: {GLOBAL_LINES},"
    f" Mean lines per patient: {GLOBAL_MEAN_LINES:.2f},"
    f" Event rate: {GLOBAL_EVENT_RATE:.2f}%,"
    f" Median PFS: {GLOBAL_MEDIAN_PFS}"
    f" (L1 Patients: {GLOBAL_L1_PATIENTS},"
    f" L1 Event rate: {GLOBAL_L1_EVENT_RATE:.2f}%,"
    f" L1 Median PFS: {GLOBAL_L1_MEDIAN_LOW}, {GLOBAL_L1_MEDIAN_PFS}, {GLOBAL_L1_MEDIAN_HIGH})"
)

anthracycline_treatments = find_line_anthracycline_treatments()
tnbc_lot1 = (
    dm.loc[
        (dm["LINE"] == 1) & (dm["HR"] == 0) & (dm["HER2"] == 0), ["PATIENT_ID", "LINE"]
    ]
    .drop_duplicates()
    .copy()
)
tnbc_lot1["LINE"] = pd.to_numeric(tnbc_lot1["LINE"], errors="coerce").astype("Int64")
anthr_line_hits = (
    anthracycline_treatments[["PATIENT_ID", "LINE"]].drop_duplicates().copy()
)
anthr_line_hits["LINE"] = pd.to_numeric(
    anthr_line_hits["LINE"], errors="coerce"
).astype("Int64")
tnbc_lot1_with_anthr = tnbc_lot1.merge(
    anthr_line_hits,
    on=["PATIENT_ID", "LINE"],
    how="left",
    indicator=True,
)
tnbc_lot1_total = int(tnbc_lot1_with_anthr.shape[0])
tnbc_lot1_hits = int((tnbc_lot1_with_anthr["_merge"] == "both").sum())
tnbc_lot1_pct = (
    float(100.0 * tnbc_lot1_hits / tnbc_lot1_total)
    if tnbc_lot1_total > 0
    else float("nan")
)
print(
    "Anthracycline exposure in TNBC mLoT1 lines: "
    f"{tnbc_lot1_hits}/{tnbc_lot1_total} ({tnbc_lot1_pct:.2f}%)"
)


first_year_line_stats = median_lines_within_first_year_from_mlot1()
print(
    "Median therapy lines started within "
    f"{first_year_line_stats['window_days']} days from modeled LoT1 start: "
    f"{first_year_line_stats['median_lines_within_window']:.2f} "
    f"(median max line reached={first_year_line_stats['median_max_line_within_window']:.2f}; "
    f"n={first_year_line_stats['n_patients']} patients)"
)
second_year_line_stats = median_lines_within_first_year_from_mlot2()
print(
    "Median therapy lines started within "
    f"{second_year_line_stats['window_days']} days from modeled LoT2 start: "
    f"{second_year_line_stats['median_lines_within_window']:.2f} "
    f"(median max line reached={second_year_line_stats['median_max_line_within_window']:.2f}; "
    f"n={second_year_line_stats['n_patients']} patients)"
)
_dm_l1_pfs = pd.to_numeric(_dm_l1["PFS_TIME_DAYS"], errors="coerce").dropna()
print(
    "mLoT1 PFS distribution check: "
    f"Q1={_dm_l1_pfs.quantile(0.25):.1f}, "
    f"median={_dm_l1_pfs.median():.1f}, "
    f"Q3={_dm_l1_pfs.quantile(0.75):.1f}, "
    f"%>365d={(_dm_l1_pfs > 365).mean() * 100:.1f}%"
)


# #
def _summarize_hr_her2_group(g: pd.DataFrame) -> pd.Series:
    median, median_low, median_high = median_pfs_ci(g["PFS_TIME_DAYS"], g["PFS_EVENT"])
    g_l1 = g[g["LINE"] == 1]
    l1_median, l1_low, l1_high = median_pfs_ci(g_l1["PFS_TIME_DAYS"], g_l1["PFS_EVENT"])

    return pd.Series(
        {
            "patients": g["PATIENT_ID"].nunique(),
            "lines": g["LINE"].count(),
            "mean_lines": g["LINE"].count() / g["PATIENT_ID"].nunique(),
            "event_rate": np.mean(g["PFS_EVENT"] == 1) * 100,
            "median_pfs": format_median_ci(median, median_low, median_high),
            "l1_patients": g_l1["PATIENT_ID"].nunique(),
            "l1_event_rate": np.mean(g_l1["PFS_EVENT"] == 1) * 100,
            "l1_median_pfs": format_median_ci(l1_median, l1_low, l1_high),
        }
    )


hr_her2_groups = (
    dm.groupby(["HR", "HER2"])
    .apply(
        _summarize_hr_her2_group,
        include_groups=False,
    )
    .reset_index(drop=True)
)
hr_her2_groups

# print the table, formatted nicely
print(
    "\nSubtype breakdown (used for `sn-article.tex`: Table `table:cohort-summary` rows):"
)
print(
    tabulate(
        hr_her2_groups,
        headers="keys",
        floatfmt=".2f",
        showindex=["HR-/HER2-", "HR-/HER2+", "HR+/HER2-", "HR+/HER2+"],
    )
)

print(
    "\nTop LoT1 planned treatment-category combinations by subtype (top 5; copy/paste friendly):"
)
print("Line 1 treatment combinations:")
treatment_combinations_to_latex_rows(dm[dm["LINE"] == 1].copy(), "TREATMENT")
# %%
dropped_lines = pd.read_csv("data/BREAST_dropped_pfs_lines.csv")
no_rad_prior = dropped_lines[
    dropped_lines["LINE_SOURCE"] == "no_radiology_within_90_prior"
]
NO_RAD_MEDIAN, NO_RAD_MEDIAN_LOW, NO_RAD_MEDIAN_HIGH = median_pfs_ci(
    no_rad_prior["ORIGINAL_PFS_TIME_DAYS"], no_rad_prior["ORIGINAL_PFS_EVENT"]
)
NO_RAD_MEDIAN_PFS = format_median_ci(
    NO_RAD_MEDIAN, NO_RAD_MEDIAN_LOW, NO_RAD_MEDIAN_HIGH
)
print(
    f"No-radiology-prior lines: {no_rad_prior.shape[0]}"
    f" from {no_rad_prior['PATIENT_ID'].nunique()} patients."
    f" event rate: {np.mean(no_rad_prior['ORIGINAL_PFS_EVENT'] == 1) * 100:.2f}%"
    f" median stats: {NO_RAD_MEDIAN_LOW}, {NO_RAD_MEDIAN_PFS}, {NO_RAD_MEDIAN_HIGH}"
)


qc_dir = OUTPUT_DIR / "extended_data"
qc_dir.mkdir(parents=True, exist_ok=True)
km_modeled_vs_no_rad = plot_km_modeled_vs_no_rad_prior(
    modeled_df=dm,
    no_rad_prior_df=no_rad_prior,
    output_path=qc_dir / "km_modeled_vs_no_rad_prior.png",
    dpi=800,
)
display(Image(filename=str(km_modeled_vs_no_rad), width=600))


# %% [markdown]
# ## Global Performance


# %%
_print_section("Model performance tables for `sn-article.tex`")
print(BASE_MODELS)


MODEL_PERF, MODEL_RESULTS_PATHS = build_model_perf(RESULTS_ROOT, BASE_MODELS)

print_formatted_model_stats(
    MODEL_PERF,
    row_order=BASE_MODELS,
)

# %%
ABLATION_CONFIGS = write_ablation_configs(
    out_path=Path("data/ablation_configs.json"),
    results_root=RESULTS_ROOT,
    models=BASE_MODELS,
    min_config_count=2,
)


# %%
surv_paths = [p / "surv_test.npz" for p in MODEL_RESULTS_PATHS[MODEL]]
shap_paths = [p / "shap_values.npz" for p in MODEL_RESULTS_PATHS[MODEL]]


SURV_OOF, SHAP_OOF, SHAP_X_OOF, SHAP_EXPECTED_OOF, TIME_GRID, FEATURES = (
    load_oof_surv_and_shap(dm=dm, surv_paths=surv_paths, shap_paths=shap_paths)
)

# %%
TIME_OF_INTEREST = [90, 180, 365, 730]
resolved_indices = [int(np.argmin(np.abs(TIME_GRID - day))) for day in TIME_OF_INTEREST]
RESOLVED_TIMES = [float(TIME_GRID[idx]) for idx in resolved_indices]
RESOLVED_TIMES_INT = [int(np.rint(t)) for t in RESOLVED_TIMES]
if RESOLVED_TIMES_INT != TIME_OF_INTEREST:
    print(
        f"Note: OOF prediction time grid resolves requested horizons {TIME_OF_INTEREST} "
        f"to {RESOLVED_TIMES_INT}."
    )
    print(
        "We report nominal horizons (e.g., 730d) in figures/tables and document the "
        "nearest-grid convention in the manuscript."
    )

# %% [markdown]
# ## Risk Stratification Figures

# %%
INTERMEDIATE_RISK_DAYS = 365
RISK_OUTPUT_DIR = Path("figures/risk")
RISK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_indices = np.arange(SURV_OOF.shape[0], dtype=int)
risk_scores = 1.0 - SURV_OOF.loc[all_indices, INTERMEDIATE_RISK_DAYS]
strata = assign_tertiles(risk_scores)
risk_scores = risk_scores.loc[strata.index]
strata_indices = {
    label: strata[strata == label].index.to_list() for label in strata.unique()
}
rmst_tau = 365


# %%
from config import RANDOM_STATE
from plots import plot_tertile_stratified_survival_and_km

PRED_CI_METHOD = "quantile"
RISK_TABLE_TIMES = [180, 365, 730]

risk_plot, risk_summary, risk_details = plot_tertile_stratified_survival_and_km(
    time_grid=TIME_GRID,
    surv_np=SURV_OOF,
    design_matrix=dm,
    output_path=RISK_OUTPUT_DIR / "tertile_stratified_survival_and_km.png",
    strata_indices=strata_indices,
    risk_scores=risk_scores,
    horizon_day=INTERMEDIATE_RISK_DAYS,
    rmst_tau=INTERMEDIATE_RISK_DAYS,
    rmst_taus=[180, 365, 730],
    risk_table_times=RISK_TABLE_TIMES,
    n_boot=400,
    random_state=RANDOM_STATE,
    cohort_label="population",
    pred_ci_method=PRED_CI_METHOD,
)

hr_plus_her2_plus_mask = (dm["HR"] == 1) & (dm["HER2"] == 1)
hr_minus_her2_plus_mask = (dm["HR"] == 0) & (dm["HER2"] == 1)
hr_plus_her2_minus_mask = (dm["HR"] == 1) & (dm["HER2"] == 0)
hr_minus_her2_minus_mask = (dm["HR"] == 0) & (dm["HER2"] == 0)

subtype_masks = {
    "HR-/HER2-": hr_minus_her2_minus_mask,
    "HR-/HER2+": hr_minus_her2_plus_mask,
    "HR+/HER2-": hr_plus_her2_minus_mask,
    "HR+/HER2+": hr_plus_her2_plus_mask,
}
group_indices = {
    label: np.where(mask)[0].tolist() for label, mask in subtype_masks.items()
}

subtype_outputs: dict[str, dict[str, object]] = {}
for subtype, mask in subtype_masks.items():
    subtype_indices = dm.index[mask].tolist()
    if not subtype_indices:
        continue
    safe_label = (
        subtype.lower().replace("/", "_").replace("+", "pos").replace("-", "neg")
    )
    subtype_plot, subtype_summary, subtype_details = (
        plot_tertile_stratified_survival_and_km(
            time_grid=TIME_GRID,
            surv_np=SURV_OOF,
            design_matrix=dm,
            output_path=RISK_OUTPUT_DIR / f"tertile_stratified_{safe_label}.png",
            strata_indices=None,
            risk_scores=risk_scores,
            subset_indices=subtype_indices,
            horizon_day=INTERMEDIATE_RISK_DAYS,
            rmst_tau=INTERMEDIATE_RISK_DAYS,
            rmst_taus=[180, 365, 730],
            risk_table_times=RISK_TABLE_TIMES,
            n_boot=400,
            random_state=RANDOM_STATE,
            cohort_label=subtype,
            show_table=True,
            pred_ci_method=PRED_CI_METHOD,
        )
    )
    subtype_outputs[subtype] = {
        "plot": subtype_plot,
        "summary": subtype_summary,
        "details": subtype_details,
        "indices": subtype_indices,
    }

lot_outputs: dict[str, dict[str, object]] = {}
lot_masks = {
    "mLoT1": dm["LINE"] == 1,
    "mLoT2+": dm["LINE"] > 1,
}
for lot_label, mask in lot_masks.items():
    lot_indices = dm.index[mask].tolist()
    if not lot_indices:
        continue
    lot_safe_label = lot_label.lower().replace("+", "plus")
    lot_plot, lot_summary, lot_details = plot_tertile_stratified_survival_and_km(
        time_grid=TIME_GRID,
        surv_np=SURV_OOF,
        design_matrix=dm,
        output_path=RISK_OUTPUT_DIR / f"tertile_stratified_{lot_safe_label}.png",
        strata_indices=None,
        risk_scores=risk_scores,
        subset_indices=lot_indices,
        horizon_day=INTERMEDIATE_RISK_DAYS,
        rmst_tau=INTERMEDIATE_RISK_DAYS,
        rmst_taus=[180, 365, 730],
        risk_table_times=RISK_TABLE_TIMES,
        n_boot=400,
        random_state=RANDOM_STATE,
        cohort_label=lot_label,
        show_table=True,
        pred_ci_method=PRED_CI_METHOD,
    )
    lot_outputs[lot_label] = {
        "plot": lot_plot,
        "summary": lot_summary,
        "details": lot_details,
        "indices": lot_indices,
    }

_print_section("mLoT tertile-stratified survival summaries")
for lot_label, lot_info in lot_outputs.items():
    print(f"\n{lot_label}:")
    print(lot_info["summary"])

# %%
_print_section("Subtype calibration summary (MAE + RMST delta)")
subtype_summary_df, pairwise_df, corr_stats = summarize_subtype_calibration(
    subtype_outputs=subtype_outputs,
    design_matrix=dm,
    surv_np=np.asarray(SURV_OOF, dtype=float),
    time_grid=np.asarray(TIME_GRID, dtype=float),
    rmst_tau=float(INTERMEDIATE_RISK_DAYS),
    n_boot=400,
    random_state=RANDOM_STATE,
    subtype_order=SUBTYPE_ORDER,
)
# print(subtype_summary_df)

# print("\nPairwise differences (bootstrap) for MAE and RMST delta:")
# print(pairwise_df)

# if corr_stats:
#     lines = corr_stats.get("n_lines", {})
#     patients = corr_stats.get("n_patients", {})
#     print(
#         f"\nSpearman correlation (n_lines vs MAE): rho={lines.get('rho', float('nan')):.3f}, p={lines.get('p', float('nan')):.3g}"
#     )
#     print(
#         f"Spearman correlation (n_patients vs MAE): rho={patients.get('rho', float('nan')):.3f}, p={patients.get('p', float('nan')):.3g}"
#     )

# %%
population_curve_path, population_metrics = plot_population_km_vs_mean(
    time_grid=TIME_GRID,
    surv_np=SURV_OOF,
    indices=all_indices,
    durations=dm[TIME_COL],
    events=dm[EVENT_COL],
    output_path=RISK_OUTPUT_DIR / "population_km_vs_mean.png",
    title="Population Kaplan–Meier vs mean prediction",
    figsize=(8.5, 5.2),
    dpi=800,
)
mae_val = population_metrics.get("mae", float("nan"))
if np.isfinite(mae_val):
    print(f"Population MAE (mean prediction vs KM): {mae_val:.3f}")
else:
    print("Population MAE unavailable.")
display(Image(filename=str(population_curve_path), width=600))

# %% [markdown]
# ## Model Performance

# %%
model_performance_dir = OUTPUT_DIR / "model_performance"
model_performance_dir.mkdir(parents=True, exist_ok=True)
all_model_paths = []
hrher_paths = []
lot_paths = []

for metric, y_limits in zip(["ipcw", "auc"], [(0.61, 0.75), (0.65, 0.95)]):
    series_data = {
        m: MODEL_PERF[m].series[metric]
        for m in BASE_MODELS
        if MODEL_PERF.get(m) and metric in MODEL_PERF[m].series
    }
    if not series_data:
        print(f"Skipping {metric} curves: no aggregated data.")
        continue
    time_points_metric = next(iter(series_data.values())).times
    fig = plot_full_group_individual_time_curves(
        models=list(series_data.keys()),
        metric=metric,
        time_points=time_points_metric,
        max_time_days=ADMIN_CENSOR_DAYS,
        shade="std",
        shade_alpha=0.18,
        y_min_auc_ipcw=0.0,
        y_limits={},
        title=f"Time-dependent {metric.upper()}",
        figsize=(9.0, 6.0),
        dpi=800,
        output_path=model_performance_dir / f"td_{metric}_full.png",
        series=series_data,
        model_name_map=DEFAULT_MODEL_NAME_MAP,
    )
    all_model_paths.append(fig)
    risk_hrher = MODEL_PERF.get(MODEL, None)
    if risk_hrher and metric in risk_hrher.hrher2_series:
        hrher_path = plot_hrher2_time_curves(
            model=MODEL,
            metric=metric,
            time_points=risk_hrher.series[metric].times,
            include_overall=False,
            shade="std",
            shade_alpha=0.18,
            y_limits=y_limits,
            title=f"{DEFAULT_MODEL_NAME_MAP.get(MODEL, MODEL)} {metric.upper()} by HR/HER2",
            figsize=(9.0, 6.0),
            dpi=800,
            output_path=model_performance_dir / f"td_{metric}_subgroup.png",
            series_overall=risk_hrher.series.get(metric),
            subgroup_series=risk_hrher.hrher2_series.get(metric, {}),
            model_name_map=DEFAULT_MODEL_NAME_MAP,
        )
        hrher_paths.append(hrher_path)
for path in all_model_paths + hrher_paths:
    display(Image(filename=str(path), width=600))

# %%
subgroup_performance = plot_hrher2_time_curves(
    model=MODEL,
    metric=metric,
    time_points=risk_hrher.series[metric].times,
    include_overall=False,
    shade="std",
    shade_alpha=0.18,
    y_limits=y_limits,
    title=f"{DEFAULT_MODEL_NAME_MAP.get(MODEL, MODEL)} {metric.upper()} by HR/HER2",
    figsize=(9.0, 6.0),
    dpi=800,
    output_path=model_performance_dir / f"td_{metric}_subgroup.png",
    series_overall=risk_hrher.series.get(metric),
    subgroup_series=risk_hrher.hrher2_series.get(metric, {}),
    model_name_map=DEFAULT_MODEL_NAME_MAP,
)
display(Image(filename=str(subgroup_performance), width=600))

# %% [markdown]
# ## Age Ablation Analysis

# %%
AGE_ABLATION_ROOT = Path("results_ablation/drop_age")
METRICS_FOR_COMPARISON: tuple[str, ...] = (
    C_INDEX,
    AUC,
    "IBS",
    "auc_at_1y",
    "auc_at_2y",
)

# Directionality for one-sided paired tests where "leakage looks better".
# For concordance/AUC metrics, higher is better -> test leakage > clean.
# For IBS, lower is better -> test leakage < clean.
AGE_METRIC_TWO_SIDED: dict[str, str] = {
    C_INDEX: "two-sided",
    AUC: "two-sided",
    "auc_at_1y": "two-sided",
    "auc_at_2y": "two-sided",
    "IBS": "two-sided",
}
AGE_METRIC_WORSE_PERFORMANCE: dict[str, str] = {
    C_INDEX: "less",
    AUC: "less",
    "auc_at_1y": "less",
    "auc_at_2y": "less",
    "IBS": "greater",
}
MODEL_PERF_AGE_ABLATION, MODEL_RESULTS_PATHS_AGE_ABLATION = build_model_perf(
    AGE_ABLATION_ROOT, BASE_MODELS
)

paired_comparison_age_ablation = paired_tests_summary(
    MODEL_PERF,
    MODEL_PERF_AGE_ABLATION,
    models=BASE_MODELS,
    metrics=METRICS_FOR_COMPARISON,
    alternative="two-sided",
    metric_alternatives=AGE_METRIC_TWO_SIDED,
)

paired_comparison_age_ablation_worse = paired_tests_summary(
    MODEL_PERF,
    MODEL_PERF_AGE_ABLATION,
    models=BASE_MODELS,
    metrics=METRICS_FOR_COMPARISON,
    alternative="less",
    metric_alternatives=AGE_METRIC_WORSE_PERFORMANCE,
)
print("AGE ABLATION MODEL RESULTS")
print("Used for `sn-article.tex`: Table `table:age-ablation-performance`.")
print_formatted_model_stats(MODEL_PERF_AGE_ABLATION, row_order=BASE_MODELS)

# %%
print("Used for `sn-article.tex`: Table `table:age-ablation-deltas`.")

delta_table_age_ablation_worse = delta_table_from_paired(
    paired_comparison_age_ablation_worse,
    metrics=(C_INDEX, AUC, "IBS"),
)
print("Used for `sn-article.tex`: Table `table:age-ablation-deltas`.")
print(delta_table_age_ablation_worse)

# %%
AGE_ABLATION_LESS_REG_ROOT = Path("results_ablation/drop_age_less_regularized")
AGE_ABLATION_LESS_REG_MODELS = [
    model
    for model in BASE_MODELS
    if (AGE_ABLATION_LESS_REG_ROOT / model / "eval_metrics.json").exists()
]
MODEL_PERF_AGE_ABLATION_LESS_REG, MODEL_RESULTS_PATHS_AGE_ABLATION_LESS_REG = (
    build_model_perf(AGE_ABLATION_LESS_REG_ROOT, AGE_ABLATION_LESS_REG_MODELS)
)

paired_comparison_age_ablation_less_reg_worse = paired_tests_summary(
    MODEL_PERF,
    MODEL_PERF_AGE_ABLATION_LESS_REG,
    models=AGE_ABLATION_LESS_REG_MODELS,
    metrics=METRICS_FOR_COMPARISON,
    alternative="less",
    metric_alternatives=AGE_METRIC_WORSE_PERFORMANCE,
)
print("AGE ABLATION (LESS REGULARIZED) MODEL RESULTS")
print("Used for extended data: age ablation with less-regularized configs.")
print_formatted_model_stats(
    MODEL_PERF_AGE_ABLATION_LESS_REG, row_order=AGE_ABLATION_LESS_REG_MODELS
)

delta_table_age_ablation_less_reg_worse = delta_table_from_paired(
    paired_comparison_age_ablation_less_reg_worse,
    metrics=(C_INDEX, AUC, "IBS"),
)
print("Used for extended data: age-ablation deltas (less-regularized, worse-test).")
print(delta_table_age_ablation_less_reg_worse)

p_signflip_age_ablation_less_reg = (
    paired_comparison_age_ablation_less_reg_worse[
        paired_comparison_age_ablation_less_reg_worse["metric"].isin((AUC, "IBS"))
    ]
    .pivot(index="model", columns="metric", values="p_signflip")
    .rename(columns={AUC: "p_signflip_AUC", "IBS": "p_signflip_IBS"})
    .reset_index()
)
print("Used for extended data: age-ablation p_signflip (less-regularized, worse-test).")
print(p_signflip_age_ablation_less_reg)
# %% [markdown]
# ## Leakage Analysis

# %%
LEAKAGE_ROOT = Path("leakage")
METRICS_FOR_COMPARISON: tuple[str, ...] = (
    C_INDEX,
    AUC,
    "IBS",
    "auc_at_1y",
    "auc_at_2y",
)

# Directionality for one-sided paired tests where "leakage looks better".
# For concordance/AUC metrics, higher is better -> test leakage > clean.
# For IBS, lower is better -> test leakage < clean.
METRIC_ALTERNATIVES: dict[str, str] = {
    C_INDEX: "greater",
    AUC: "greater",
    "auc_at_1y": "greater",
    "auc_at_2y": "greater",
    "IBS": "less",
}
MODEL_PERF_LEAKAGE, MODEL_RESULTS_PATHS_LEAKAGE = build_model_perf(
    LEAKAGE_ROOT,
    LEAKAGE_MODELS,
)


paired_comparison = paired_tests_summary(
    MODEL_PERF,
    MODEL_PERF_LEAKAGE,
    models=LEAKAGE_MODELS,
    metrics=METRICS_FOR_COMPARISON,
    alternative="greater",
    metric_alternatives=METRIC_ALTERNATIVES,
)
print("LEAKAGE MODEL RESULTS")
print("Used for `sn-article.tex`: Table `table:leakage-performance`.")
print_formatted_model_stats(MODEL_PERF_LEAKAGE, row_order=LEAKAGE_MODELS)


# %%
delta_table = delta_table_from_paired(
    paired_comparison,
    metrics=(C_INDEX, AUC, "IBS"),
)
print("Used for `sn-article.tex`: Table `table:leakage-deltas`.")
display(delta_table)


surv_paths_leakage = [
    path / "surv_test.npz" for path in MODEL_RESULTS_PATHS_LEAKAGE[MODEL]
]
# update the first part to 'leakage' instead
for i in range(len(surv_paths_leakage)):
    parts = surv_paths_leakage[i].parts
    new_parts = (LEAKAGE_ROOT,) + parts[1:]
    surv_paths_leakage[i] = Path(*new_parts)


surv_leakage_df, time_leakage = load_oof_surv_only(dm, surv_paths_leakage)
assert TIME_GRID.shape == time_leakage.shape
assert np.allclose(TIME_GRID, time_leakage)


# %%
from sksurv.util import Surv
from dl_lib.pycox.evaluation.eval_surv import EvalSurv
from utils import compute_time_dependent_auc, time_dependent_evals
from SurvivalEVAL import SurvivalEvaluator


def eval_within_manuscript(
    surv_df: pd.DataFrame,
    durations: pd.Series,
    events: pd.Series,
    time_grid: np.ndarray,
    eval_horizons: Sequence[float],
):

    eval_pycox = EvalSurv(
        surv=surv_df.T,
        durations=durations.to_numpy(dtype=float),
        events=events.to_numpy(dtype=bool),
    )
    C_index = float(eval_pycox.concordance_td("adj_antolini"))

    auc_info = compute_time_dependent_auc(
        surv_test_np=surv_df.to_numpy(),
        time_grid_train_np=time_grid,
        y_train=Surv.from_arrays(
            event=events.to_numpy(dtype=bool),
            time=durations.to_numpy(dtype=float),
        ),
        y_test=Surv.from_arrays(
            event=events.to_numpy(dtype=bool),
            time=durations.to_numpy(dtype=float),
        ),
        eval_horizons=eval_horizons,
    )

    surv_evaluator = SurvivalEvaluator(
        pred_survs=surv_df.to_numpy(),
        time_coordinates=time_grid,
        test_event_times=durations.to_numpy(dtype=float),
        test_event_indicators=events.to_numpy(dtype=bool),
        train_event_times=durations.to_numpy(dtype=float),
        train_event_indicators=events.to_numpy(dtype=bool),
    )

    (
        eval_time_indices,
        ipcw,
        brier_score,
        IBS,
        p_value,
        weighted_mae_margin,
        weighted_mae_po,
    ) = time_dependent_evals(
        eval_times=eval_horizons,
        time_grid_train_np=time_grid,
        y_train=Surv.from_arrays(
            event=events.to_numpy(dtype=bool),
            time=durations.to_numpy(dtype=float),
        ),
        y_test=Surv.from_arrays(
            event=events.to_numpy(dtype=bool),
            time=durations.to_numpy(dtype=float),
        ),
        surv_test_np=surv_df.to_numpy(dtype=float),
        surv_evaluator=surv_evaluator,
    )
    return {
        "C_index": C_index,
        "auc_info": auc_info,
        "ipcw": ipcw,
        "brier_score": brier_score,
        "IBS": IBS,
        "p_value": p_value,
        "weighted_mae_margin": weighted_mae_margin,
        "weighted_mae_po": weighted_mae_po,
    }


# %%
planned_cols = [c for c in dm.columns if c.startswith("PLANNED_")]
planned_suffixes = [c.replace("PLANNED_", "") for c in planned_cols]
received_cols = [c for c in dm.columns if c.startswith("RECEIVED_")]
# there is an IMMUNO_OTHER column that has no PLANNED counterpart; ignore it first
received_cols = [c for c in received_cols if c != "RECEIVED_AGENT_IMMUNO_OTHER"]
received_suffixes = [c.replace("RECEIVED_", "") for c in received_cols]
assert set(planned_suffixes) == set(received_suffixes)
# get indices in dm where any values of planned and received for the same suffix differ
has_late_start_agent = np.zeros(dm.shape[0], dtype=bool)
# assert only 1 agent was received
assert dm["RECEIVED_AGENT_IMMUNO_OTHER"].sum() == 1
# that row should be included in late-start agents
has_late_start_agent = has_late_start_agent | (
    dm["RECEIVED_AGENT_IMMUNO_OTHER"] == 1
).to_numpy(dtype=bool)
for suffix in planned_suffixes:
    planned_col = f"PLANNED_{suffix}"
    received_col = f"RECEIVED_{suffix}"
    diff_mask = dm[planned_col] != dm[received_col]
    has_late_start_agent = has_late_start_agent | diff_mask

has_late_start_systemic_agent = np.zeros(dm.shape[0], dtype=bool)
has_late_start_systemic_agent = has_late_start_systemic_agent | (
    dm["RECEIVED_AGENT_IMMUNO_OTHER"] == 1
).to_numpy(dtype=bool)
for suffix in planned_suffixes:
    if not any(x in suffix for x in ["SURGERY", "RADIATION_THERAPY", "BONE TREATMENT"]):
        planned_col = f"PLANNED_{suffix}"
        received_col = f"RECEIVED_{suffix}"
        diff_mask = dm[planned_col] != dm[received_col]
        has_late_start_systemic_agent = has_late_start_systemic_agent | diff_mask
    else:
        print(f"Skipping non-systemic agent for systemic mask: {suffix}")

late_start_indices = np.where(has_late_start_agent)[0].tolist()
late_start_systemic_indices = np.where(has_late_start_systemic_agent)[0].tolist()
late_start_local_indices = sorted(
    set(late_start_indices) - set(late_start_systemic_indices)
)
print(f"Number of patients with late-start agents: {len(late_start_indices)}")
print(
    f"Number of patients with late-start systemic agents: {len(late_start_systemic_indices)}"
)
print(
    f"Number of patients with late-start local agents: {len(late_start_local_indices)}"
)

# %%
late_start_dir = OUTPUT_DIR / "extended_data"
late_start_dir.mkdir(parents=True, exist_ok=True)

km_path = plot_km_late_start_vs_rest(
    dm=dm,
    late_start_indices=late_start_indices,
    output_path=late_start_dir / "KM_late_start.png",
    dpi=800,
)
display(Image(filename=str(km_path), width=600))

systemic_local_path, systemic_local_mae = (
    plot_mean_surv_late_start_systemic_vs_local_clean_leakage(
        dm=dm,
        time_grid=TIME_GRID,
        surv_clean=SURV_OOF,
        surv_leakage=surv_leakage_df,
        late_start_systemic_indices=late_start_systemic_indices,
        late_start_local_indices=late_start_local_indices,
        output_path=late_start_dir / "mean_surv_systemic_vs_local_clean_leakage.png",
        dpi=800,
    )
)
print(
    f"MAE systemic clean vs KM: {systemic_local_mae['systemic_clean_vs_km']:.6g}; "
    f"systemic leakage vs KM: {systemic_local_mae['systemic_leakage_vs_km']:.6g}"
)
print(
    f"MAE local clean vs KM: {systemic_local_mae['local_clean_vs_km']:.6g}; "
    f"local leakage vs KM: {systemic_local_mae['local_leakage_vs_km']:.6g}"
)
display(Image(filename=str(systemic_local_path), width=600))

mean_path, mae = plot_mean_surv_late_start(
    dm=dm,
    time_grid=TIME_GRID,
    surv_clean=SURV_OOF,
    surv_leakage=surv_leakage_df,
    late_start_indices=late_start_indices,
    output_path=late_start_dir / "mean_surv_late_start.png",
    dpi=800,
)
print(f"MAE (rest clean vs KM): {mae['rest_clean_vs_km']:.6g}")
print(f"MAE (rest leakage vs KM): {mae['rest_leakage_vs_km']:.6g}")
print(f"MAE (late clean vs KM): {mae['late_clean_vs_km']:.6g}")
print(f"MAE (late leakage vs KM): {mae['late_leakage_vs_km']:.6g}")
display(Image(filename=str(mean_path), width=600))


# %%
print_time_dependent_auc_and_c(MODEL_PERF)

# %% [markdown]
# ## Ablation Experiments

# %%
from config import MODALITY_GROUPS

MODALITY_PERF = {}


def _add_scalar(name: str, values: list[float]):
    scalars[name] = ScalarMetricSummary(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        values=[float(v) for v in values],
    )


for modality in MODALITY_GROUPS.keys():
    MODALITY_STAT = {"only": {}, "exclude": {}}
    for ablation_type in ("only", "exclude"):
        results_path = f"results_ablation/{ablation_type}_{modality}"
        within_modality_stats: Dict[str, Any] = {}
        for model in ABLATION_MODELS:
            folds = OuterFolds.from_json(
                json.load(open(Path(results_path) / model / "eval_metrics.json"))[
                    "outer_folds"
                ]
            )
            scalars: Dict[str, ScalarMetricSummary] = {}

            vals_c = []
            vals_auc = []
            for _, fold in folds.items():
                fm = fold.metrics
                vals_c.append(fm[C_INDEX])
                vals_auc.append(fm[AUC])

            _add_scalar(C_INDEX, vals_c)
            _add_scalar(AUC, vals_auc)
            within_modality_stats[model] = ModelStat(
                name=model,
                scalars=scalars,
                series={},
                hrher2_series={},
            )
        MODALITY_STAT[ablation_type] = within_modality_stats
        MODALITY_STAT[ablation_type]["best_model"] = max(
            within_modality_stats.items(),
            key=lambda item: (
                round(item[1].scalars[C_INDEX].mean, 3),
                item[1].scalars[AUC].mean,
            ),
        )[0]

    MODALITY_PERF[modality] = MODALITY_STAT


# %%
from plots import plot_modality_best_model_metric_deltas

ablation_only = plot_modality_best_model_metric_deltas(
    modality_perf=MODALITY_PERF,
    baseline_stat=MODEL_PERF[MODEL],
    ablation_type="only",
    figsize=(8.5, 6.0),
    dpi=800,
    output_path=OUTPUT_DIR / "model_performance" / "ablation_only.png",
    model_name_map=DEFAULT_MODEL_NAME_MAP,
)
display(Image(filename=str(ablation_only), width=600))
ablation_exclude = plot_modality_best_model_metric_deltas(
    modality_perf=MODALITY_PERF,
    baseline_stat=MODEL_PERF[MODEL],
    ablation_type="exclude",
    figsize=(8.5, 6.0),
    dpi=800,
    output_path=OUTPUT_DIR / "model_performance" / "ablation_exclude.png",
    model_name_map=DEFAULT_MODEL_NAME_MAP,
)
display(Image(filename=str(ablation_exclude), width=600))

# %%
age_ablation_stats: Dict[str, Any] = {}
for model in BASE_MODELS:
    folds = OuterFolds.from_json(
        json.load(
            open(Path("results_ablation/drop_age") / model / "eval_metrics.json")
        )["outer_folds"]
    )
    scalars: Dict[str, ScalarMetricSummary] = {}

    vals_c = []
    vals_auc = []
    for _, fold in folds.items():
        fm = fold.metrics
        vals_c.append(fm[C_INDEX])
        vals_auc.append(fm[AUC])

    _add_scalar(C_INDEX, vals_c)
    _add_scalar(AUC, vals_auc)
    age_ablation_stats[model] = ModelStat(
        name=model,
        scalars=scalars,
        series={},
        hrher2_series={},
    )
MODALITY_STAT["age"] = age_ablation_stats
MODALITY_STAT["age"]["best_model"] = max(
    age_ablation_stats.items(),
    key=lambda item: (
        round(item[1].scalars[C_INDEX].mean, 3),
        item[1].scalars[AUC].mean,
    ),
)[0]
MODALITY_PERF["age"] = MODALITY_STAT

# %% [markdown]
# ## Explanability Results
#

# %%
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


from plots import load_feature_group_map

SHAP_DIR = OUTPUT_DIR / "feature_importance"
FEATURE_LABELS = json.load(open("data/feature_display_names.json"))
group_map = load_feature_group_map(
    feature_names=FEATURES, groups_path=Path("data/feature_groups.json")
)
group_mean_path = plot_group_mean_abs_shap(
    shap_values=SHAP_OOF,
    feature_names=FEATURES,
    group_map=group_map,
    label_map=FEATURE_LABELS,
    top_k=10,
    title="Feature-group contribution share to 365d progression risk",
    figsize=(8.0, 5.0),
    dpi=800,
    output_path=SHAP_DIR / "shap_group_mean.png",
)
display(Image(filename=str(group_mean_path), width=600))


# %%
# For SHAP visualization semantics, use 365-day progression risk (1 - survival).
# This flips SHAP direction so red corresponds to higher predicted risk.
SHAP_RISK_OOF = -SHAP_OOF
SHAP_EXPECTED_RISK_OOF = 1.0 - SHAP_EXPECTED_OOF

shap_summary_path = plot_shap_summary_from_arrays(
    shap_values=SHAP_RISK_OOF,
    X=SHAP_X_OOF,
    feature_names=FEATURES,
    label_map=FEATURE_LABELS,
    top_k=12,
    title="Top 12 SHAP features",
    figsize=(9.0, 5.0),
    dpi=800,
    output_path=SHAP_DIR / "shap_individual.png",
)

display(Image(filename=str(shap_summary_path), width=600))

from typing import Dict, List, Union

import numpy as np
import pandas as pd

IndexLike = Union[pd.Index, List, np.ndarray]


def _top_k_strings(
    smd: pd.Series,
    label_map: Dict[str, str] | None,
    top_k: int = 5,
) -> list[str]:
    if smd.empty:
        return []
    rows = []
    top_names = smd.abs().sort_values(ascending=False).head(top_k).index
    for name in top_names:
        val = smd[name]
        label = label_map.get(name, name) if label_map else name
        rows.append(f"{label} ({val:+.3f})")
    return rows


def _top_k_by_sign(
    smd: pd.Series,
    label_map: Dict[str, str] | None,
    top_k: int = 5,
    sign: str = "pos",
) -> list[str]:
    if smd.empty:
        return []
    if sign == "pos":
        subset = smd[smd > 0].sort_values(ascending=False)
    elif sign == "neg":
        subset = smd[smd < 0].sort_values(ascending=True)
    else:
        raise ValueError("sign must be 'pos' or 'neg'")
    rows = []
    for name, val in subset.head(top_k).items():
        label = label_map.get(name, name) if label_map else name
        rows.append(f"{label} ({val:+.3f})")
    return rows


def _pad_rows(rows: list[str], n: int) -> list[str]:
    if len(rows) < n:
        return rows + [""] * (n - len(rows))
    return rows[:n]


def _print_single_col_table(
    rows: list[str], col_name: str, min_rows: int = 5
) -> None:
    if not rows:
        rows = ["(no data)"]
    n = max(len(rows), min_rows)
    rows = _pad_rows(rows, n)
    df = pd.DataFrame({col_name: rows})
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))


def _print_two_col_table(
    left_rows: list[str],
    right_rows: list[str],
    left_name: str,
    right_name: str,
    min_rows: int = 5,
) -> None:
    n = max(len(left_rows), len(right_rows), min_rows)
    left_rows = _pad_rows(left_rows, n)
    right_rows = _pad_rows(right_rows, n)
    df = pd.DataFrame({left_name: left_rows, right_name: right_rows})
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))


def _split_artifacts(
    smd: pd.Series, threshold: float = 5.0
) -> tuple[pd.Series, pd.Series]:
    if smd.empty:
        return smd, smd
    mask = smd.abs() > threshold
    artifacts = smd[mask]
    clean = smd[~mask]
    return clean, artifacts


def _print_artifacts(
    artifacts: pd.Series,
    label_map: Dict[str, str] | None,
    title: str,
) -> None:
    if artifacts.empty:
        return
    ordered = artifacts.reindex(artifacts.abs().sort_values(ascending=False).index)
    rows = []
    for name, val in ordered.items():
        label = label_map.get(name, name) if label_map else name
        rows.append({"Artifact": label, "d (H--L)": f"{val:+.3f}"})
    print(f"\n{title}")
    print(tabulate(pd.DataFrame(rows), headers="keys", tablefmt="github", showindex=False))




hr_plus_her2_plus_mask = (dm["HR"] == 1) & (dm["HER2"] == 1)
hr_minus_her2_plus_mask = (dm["HR"] == 0) & (dm["HER2"] == 1)
hr_plus_her2_minus_mask = (dm["HR"] == 1) & (dm["HER2"] == 0)
hr_minus_her2_minus_mask = (dm["HR"] == 0) & (dm["HER2"] == 0)

subtype_masks = {
    "HR-/HER2-": hr_minus_her2_minus_mask,
    "HR-/HER2+": hr_minus_her2_plus_mask,
    "HR+/HER2-": hr_plus_her2_minus_mask,
    "HR+/HER2+": hr_plus_her2_plus_mask,
}

# Build global high/low masks aligned to SHAP outputs
high_mask_all = SHAP_OOF.index.isin(risk_details["strata_indices"]["high"])
low_mask_all = SHAP_OOF.index.isin(risk_details["strata_indices"]["low"])

line_values_for_contrast = pd.to_numeric(
    dm.loc[SHAP_X_OOF.index, "LINE"], errors="coerce"
)
mlot1_mask = (line_values_for_contrast == 1).to_numpy()
mlot2plus_mask = (line_values_for_contrast > 1).to_numpy()


def _align_mask(mask: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(mask, pd.Series):
        return mask.reindex(SHAP_X_OOF.index).fillna(False).to_numpy()
    return np.asarray(mask, dtype=bool)


cohort_defs = [
    ("HR-/HER2-", _align_mask(subtype_masks["HR-/HER2-"])),
    ("HR-/HER2+", _align_mask(subtype_masks["HR-/HER2+"])),
    ("HR+/HER2-", _align_mask(subtype_masks["HR+/HER2-"])),
    ("HR+/HER2+", _align_mask(subtype_masks["HR+/HER2+"])),
    ("mLoT1", mlot1_mask),
    ("mLoT2+", mlot2plus_mask),
]

print("\n-------- Baseline feature value contrasts (centered, H--L) --------")
for cohort_name, cohort_mask in cohort_defs:
    print(f"\n{cohort_name}")
    smd_feat = baseline_feature_value_contrast(
        SHAP_X_OOF,
        high_sel=high_mask_all & cohort_mask,
        low_sel=low_mask_all & cohort_mask,
        cohort_sel=cohort_mask,
    )
    smd_feat, artifacts = _split_artifacts(smd_feat, threshold=5.0)
    _print_artifacts(
        artifacts,
        FEATURE_LABELS,
        "Artifacts (|d| > 5) excluded from baseline ranking",
    )
    rows = _top_k_strings(smd_feat, FEATURE_LABELS, top_k=5)
    _print_single_col_table(rows, "Baseline feature (d, H--L)")

print("\n-------- SHAP group share contrasts (H--L) --------")
for cohort_name, cohort_mask in cohort_defs:
    print(f"\n{cohort_name}")
    shap_feat = baseline_feature_shap_contrast(
        SHAP_RISK_OOF,
        high_sel=high_mask_all & cohort_mask,
        low_sel=low_mask_all & cohort_mask,
        cohort_sel=cohort_mask,
    )
    shap_group = group_feature_contrast(
        SHAP_RISK_OOF,
        group_map=group_map,
        high_sel=high_mask_all & cohort_mask,
        low_sel=low_mask_all & cohort_mask,
        cohort_sel=cohort_mask,
    )
    shap_feat, feat_artifacts = _split_artifacts(shap_feat, threshold=5.0)
    shap_group, group_artifacts = _split_artifacts(shap_group, threshold=5.0)
    _print_artifacts(
        feat_artifacts,
        FEATURE_LABELS,
        "Artifacts (|d| > 5) excluded from SHAP feature ranking",
    )
    _print_artifacts(
        group_artifacts,
        None,
        "Artifacts (|d| > 5) excluded from SHAP group ranking",
    )
    feat_rows = _top_k_by_sign(shap_feat, FEATURE_LABELS, top_k=5, sign="pos")
    _print_single_col_table(feat_rows, "Feature |phi| (d, H--L; positive only)")
    group_rows = _top_k_strings(shap_group, None, top_k=5)
    _print_single_col_table(group_rows, "Group share (d, H--L)")


# %%
LEFT_IDX = 7228
waterfall_path_left = plot_single_sample_shap_waterfall_from_arrays(
    shap_values=SHAP_RISK_OOF.loc[LEFT_IDX].to_numpy(),
    X=SHAP_X_OOF.loc[LEFT_IDX].to_numpy(),
    expected_value=float(SHAP_EXPECTED_RISK_OOF.loc[LEFT_IDX]),
    feature_names=FEATURES,
    label_map=FEATURE_LABELS,
    title="Example low-risk population patient (365d progression risk)",
    max_display=8,
    figsize=(9.0, 6.0),
    dpi=800,
    output_path=SHAP_DIR / "waterfall_left.png",
)
display(Image(filename=str(waterfall_path_left), width=600))

# %%
RIGHT_IDX = 5128
waterfall_path_right = plot_single_sample_shap_waterfall_from_arrays(
    shap_values=SHAP_RISK_OOF.loc[RIGHT_IDX].to_numpy(),
    X=SHAP_X_OOF.loc[RIGHT_IDX].to_numpy(),
    expected_value=float(SHAP_EXPECTED_RISK_OOF.loc[RIGHT_IDX]),
    feature_names=FEATURES,
    label_map=FEATURE_LABELS,
    title="Example high-risk population patient (365d progression risk)",
    max_display=7,
    figsize=(9.0, 6.0),
    dpi=800,
    output_path=SHAP_DIR / "waterfall_right.png",
)
display(Image(filename=str(waterfall_path_right), width=600))


# %% [markdown]
# ## Extended Data tables

# %%
_print_section(
    "Treatment-combination tables for `sn-article.tex` (`table:subtype-top-tx`, `table:subtype-top-tx-lot1`)"
)
print("mLoT2+ combinations:")
treatment_combinations_to_latex_rows(dm[dm["LINE"] != 1], "TREATMENT")
print("Line 1 drug combinations:")
treatment_combinations_to_latex_rows(dm[dm["LINE"] == 1].copy(), "AGENT")
# %%

dm_summary = dm.groupby(["PATIENT_ID"]).agg("first")
age_mean = dm_summary["AGE"].mean()
age_std = dm_summary["AGE"].std()
n_patients = dm_summary.shape[0]
_print_section("Baseline summary for `sn-article.tex`: Table `table:baseline`")
print(f"Age: {age_mean:.1f} ± {age_std:.1f} years (n={n_patients})")
print(f"hr positive: {dm_summary['HR'].sum()} ({dm_summary['HR'].mean() * 100:.1f}%)")
print(
    f"her2 positive: {dm_summary['HER2'].sum()} ({dm_summary['HER2'].mean() * 100:.1f}%)"
)
print(
    f"stage iv: {dm_summary['STAGE_CDM_DERIVED_IV'].sum()} ({dm_summary['STAGE_CDM_DERIVED_IV'].mean() * 100:.1f}%)"
)
print(
    f"stage i-iii : {dm_summary.shape[0] - dm_summary['STAGE_CDM_DERIVED_IV'].sum()} ({(1 - dm_summary['STAGE_CDM_DERIVED_IV'].mean()) * 100:.1f}%)"
)
# get top 5 genomics alterations
genomics_cols = [c for c in dm.columns if c.startswith("GENOMICS_")]
genomics_sums = dm_summary[genomics_cols].sum().sort_values(ascending=False)
print("Top 5 genomic alterations:")
for gene, count in genomics_sums.head(5).items():
    print(f"  {gene}: {count} ({count / n_patients * 100:.1f}%)")

# %%
AGE_BINS = [0, 40, 50, 65, 75, 200]
AGE_LABELS = [r"$\leq$40", "41--50", "51--65", "66--75", "76+"]
print("\nUsed for `sn-article.tex`: Table `table:hr-her2-by-age`.")
print_hr_her2_by_age_midrule_block(dm)
