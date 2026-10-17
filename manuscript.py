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

import sys
import traceback
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from config import ADMIN_CENSOR_DAYS, AUC, C_INDEX, EVENT_COL, FIGURES_PATH, RANDOM_STATE, RESULTS_PATH, TIME_COL
from manuscript_utils import (BASE_MODELS, DEFAULT_MODEL_NAME_MAP, MANUSCRIPT_TIME_HORIZONS, SUBTYPE_ORDER,
                              ManuscriptContext, build_group_shap_share_df, format_median_ci, genomic_masking,
                              horizon_summary, mean_sd, median_pfs_ci, tertile_strata)
from plots import (plot_full_group_individual_time_curves, plot_group_mean_abs_shap, plot_hrher2_time_curves,
                   plot_km_late_start_vs_rest, plot_km_modeled_vs_no_rad_prior, plot_mean_surv_late_start,
                   plot_mean_surv_late_start_systemic_vs_local_clean_leakage, plot_modality_best_model_metric_deltas,
                   plot_population_km_vs_mean, plot_shap_summary_from_arrays,
                   plot_single_sample_shap_waterfall_from_arrays, plot_tertile_stratified_survival_and_km)
from paper_tables import Context, load, render
from paper_tables.common import WIDTH

MANUSCRIPT_RISK_TABLE_TIMES: tuple[int, ...] = (180, 365, 730)
MANUSCRIPT_WATERFALL_LEFT_IDX = 7228
MANUSCRIPT_WATERFALL_RIGHT_IDX = 5128
PERFORMANCE_DIR = FIGURES_PATH / "model_performance"
RISK_DIR = FIGURES_PATH / "risk"
SHAP_DIR = FIGURES_PATH / "feature_importance"
EXTENDED_DIR = FIGURES_PATH / "extended_data"


def print_markdown_table(df: pd.DataFrame, *, showindex: bool = False) -> None:
    if df.empty:
        print("(no rows)")
        return
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=showindex))


def print_stratum_mae_table(summary_df: pd.DataFrame) -> None:
    mae_df = summary_df[["stratum", "mae_pred_vs_km"]].rename(
        columns={"stratum": "Stratum", "mae_pred_vs_km": "MAE (mean prediction vs KM)"})
    mae_df["MAE (mean prediction vs KM)"] = mae_df["MAE (mean prediction vs KM)"].map(
        lambda value: f"{float(value):.3f}" if np.isfinite(value) else "nan")
    print("Stratum MAE:")
    print_markdown_table(mae_df)


def print_time_grid_resolution_note(time_grid: np.ndarray, requested_horizons=MANUSCRIPT_TIME_HORIZONS) -> None:
    time_grid = np.asarray(time_grid, dtype=float)
    resolved = [int(np.rint(float(time_grid[int(np.argmin(np.abs(time_grid - day)))]))) for day in requested_horizons]
    if list(requested_horizons) != resolved:
        print(f"Nearest-grid evaluation note: requested horizons {list(requested_horizons)} resolve to {resolved}.")


def print_time_dependent_metric_table(model_perf, metric_key: str) -> None:
    stats = horizon_summary({DEFAULT_MODEL_NAME_MAP[m]: stat.series[metric_key] for m, stat in model_perf.items()})
    print_markdown_table(mean_sd(stats).rename(columns="{}d".format).rename_axis("Model").reset_index())


def summarize_group_rows(dm: pd.DataFrame, indices, *, label: str) -> dict[str, str]:
    group_df = dm.iloc[list(indices)]
    median, low, high = median_pfs_ci(group_df[TIME_COL], group_df[EVENT_COL])
    return {"Group": label, "mLoTs": f"{group_df.shape[0]}", "Patients": f"{group_df['PATIENT_ID'].nunique()}",
            "Event rate (%)": f"{float(np.mean(group_df[EVENT_COL] == 1) * 100):.2f}",
            "Median PFS": format_median_ci(median, low, high, digits=0)}


def plot_stratified(ctx: ManuscriptContext, output_name: str, strata, cohort_label: str, subset_indices=None):
    return plot_tertile_stratified_survival_and_km(
        time_grid=ctx.time_grid, surv_np=ctx.surv_oof, design_matrix=ctx.dm, output_path=RISK_DIR / output_name,
        strata_indices=strata, risk_scores=ctx.risk_scores, subset_indices=subset_indices, horizon_day=ctx.risk_day,
        rmst_tau=ctx.risk_day, rmst_taus=MANUSCRIPT_RISK_TABLE_TIMES, risk_table_times=MANUSCRIPT_RISK_TABLE_TIMES,
        n_boot=400, random_state=RANDOM_STATE, cohort_label=cohort_label)


def emit_fig_perf_and_risk(ctx: ManuscriptContext) -> None:
    for metric in ("ipcw", "auc"):
        series = {model: ctx.model_perf[model].series[metric] for model in BASE_MODELS}
        plot_full_group_individual_time_curves(
            models=list(series), metric=metric, time_points=next(iter(series.values())).times,
            max_time_days=ADMIN_CENSOR_DAYS, shade="std", shade_alpha=0.18, title=f"Time-dependent {metric.upper()}",
            figsize=(9.0, 6.0), dpi=800, output_path=PERFORMANCE_DIR / f"td_{metric}_full.png", series=series, model_name_map=DEFAULT_MODEL_NAME_MAP)
    print("subfigure: fig:time-dependent-perf")
    print_time_grid_resolution_note(ctx.time_grid)
    print("Time-dependent IPCW C-index (mean ± SD):")
    print_time_dependent_metric_table(ctx.model_perf, "ipcw")
    print("\nTime-dependent AUC (mean ± SD):")
    print_time_dependent_metric_table(ctx.model_perf, "auc")

    print("\nsubfigure: fig:risk-strat")
    _, risk_summary, _ = plot_stratified(ctx, "tertile_stratified_survival_and_km.png", ctx.risk_strata, "population")
    print("\nPopulation tertile summary:")
    print(risk_summary.to_string(index=False))
    print_stratum_mae_table(risk_summary)
    _, population_metrics = plot_population_km_vs_mean(
        time_grid=ctx.time_grid, surv_np=ctx.surv_oof, indices=np.arange(ctx.surv_oof.shape[0], dtype=int),
        durations=ctx.dm[TIME_COL], events=ctx.dm[EVENT_COL], output_path=RISK_DIR / "population_km_vs_mean.png",
        title="Population Kaplan-Meier vs mean prediction", figsize=(8.5, 5.2), dpi=800)
    mae_val = population_metrics.get("mae", float("nan"))
    if np.isfinite(mae_val):
        print(f"Population MAE (mean prediction vs KM): {mae_val:.3f}")


def emit_fig_stratify_by_subtype(ctx: ManuscriptContext) -> None:
    for subtype in SUBTYPE_ORDER:
        indices = ctx.dm.index[ctx.subtype_masks[subtype]].tolist()
        safe_label = subtype.lower().replace("/", "_").replace("+", "pos").replace("-", "neg")
        print(subtype)
        _, summary, _ = plot_stratified(ctx, f"tertile_stratified_{safe_label}.png",
                                        tertile_strata(ctx.risk_scores, indices), f"{subtype} (local tertiles)", indices)
        print(summary.to_string(index=False))
        print_stratum_mae_table(summary)
        print()
    for lot_label, mask in ctx.lot_masks.items():
        indices = ctx.dm.index[mask].tolist()
        print(lot_label)
        _, summary, _ = plot_stratified(ctx, f"tertile_stratified_{lot_label.lower().replace('+', 'plus')}.png",
                                        tertile_strata(ctx.risk_scores, indices), f"{lot_label} (local tertiles)", indices)
        print(summary.to_string(index=False))
        print_stratum_mae_table(summary)
        print()


def emit_fig_shap_compact(ctx: ManuscriptContext) -> None:
    plot_shap_summary_from_arrays(
        shap_values=ctx.shap_risk_oof, X=ctx.shap_x_oof, feature_names=ctx.features, label_map=ctx.feature_labels,
        top_k=12, title="Top 12 SHAP features", figsize=(9.0, 5.0), dpi=800, output_path=SHAP_DIR / "shap_individual.png")
    print("subfigure: fig:shap-individual")
    mean_abs = ctx.shap_risk_oof.abs().mean(axis=0).sort_values(ascending=False).head(12)
    print_markdown_table(pd.DataFrame({"Feature": [ctx.feature_labels.get(name, name) for name in mean_abs.index],
                                       "Mean |SHAP|": [f"{float(value):.4f}" for value in mean_abs.to_numpy()]}))
    for label, row_idx, max_display, output_name in (
            ("fig:patient-example-0", MANUSCRIPT_WATERFALL_LEFT_IDX, 8, "waterfall_left.png"),
            ("fig:patient-example-1", MANUSCRIPT_WATERFALL_RIGHT_IDX, 7, "waterfall_right.png")):
        shap_values, feature_values = ctx.shap_risk_oof.loc[row_idx], ctx.shap_x_oof.loc[row_idx]
        base_risk = float(ctx.shap_expected_risk_oof.loc[row_idx])
        plot_single_sample_shap_waterfall_from_arrays(
            shap_values=shap_values.to_numpy(), X=feature_values.to_numpy(), expected_value=base_risk,
            feature_names=ctx.features, label_map=ctx.feature_labels, title=label, max_display=max_display,
            figsize=(9.0, 6.0), dpi=800, output_path=SHAP_DIR / output_name)
        print(f"\nsubfigure: {label}")
        print(f"Sample index {row_idx}: base risk={base_risk:.4f}, "
              f"predicted 365d progression risk={float(ctx.shap_expected_risk_oof.loc[row_idx] + shap_values.sum()):.4f}")
        order = shap_values.abs().sort_values(ascending=False).head(max_display).index
        print_markdown_table(pd.DataFrame([{"Feature": ctx.feature_labels.get(name, name),
                                            "Feature value": f"{float(feature_values.loc[name]):.4g}",
                                            "SHAP risk contribution": f"{float(shap_values.loc[name]):+.4f}"}
                                           for name in order]))


def emit_fig_ablation(ctx: ManuscriptContext, ablation_type: str) -> None:
    baseline = ctx.model_perf[ctx.model]
    plot_modality_best_model_metric_deltas(
        modality_perf=ctx.modality_perf, baseline_stat=baseline, ablation_type=ablation_type, figsize=(8.5, 6.0),
        dpi=800, output_path=PERFORMANCE_DIR / f"ablation_{ablation_type}.png", model_name_map=DEFAULT_MODEL_NAME_MAP)
    rows = []
    for modality in sorted(ctx.modality_perf):
        entry = ctx.modality_perf[modality].get(ablation_type)
        if not entry:
            continue
        stats = entry[entry["best_model"]]
        c_mean, auc_mean = float(stats.scalars[C_INDEX].mean), float(stats.scalars[AUC].mean)
        rows.append({"Modality": modality, "Best model": DEFAULT_MODEL_NAME_MAP[entry["best_model"]],
                     "C-index": f"{c_mean:.3f}", "ΔC": f"{c_mean - float(baseline.scalars[C_INDEX].mean):+.3f}",
                     "Mean AUC": f"{auc_mean:.3f}", "ΔAUC": f"{auc_mean - float(baseline.scalars[AUC].mean):+.3f}"})
    print_markdown_table(pd.DataFrame(rows))


def emit_fig_ablations(ctx: ManuscriptContext) -> None:
    for ablation_type in ("only", "exclude"):
        print(("\n" if ablation_type == "exclude" else "") + f"subfigure: fig:ablation-{ablation_type}")
        emit_fig_ablation(ctx, ablation_type)


def emit_fig_km_diff_preds(ctx: ManuscriptContext) -> None:
    late_start = ctx.late_start["late_start_indices"]
    systemic = ctx.late_start["late_start_systemic_indices"]
    local = ctx.late_start["late_start_local_indices"]
    plot_km_late_start_vs_rest(dm=ctx.dm, late_start_indices=late_start, dpi=800,
                               output_path=EXTENDED_DIR / "KM_late_start.png")
    _, systemic_local_mae = plot_mean_surv_late_start_systemic_vs_local_clean_leakage(
        dm=ctx.dm, time_grid=ctx.time_grid, surv_clean=ctx.surv_oof, surv_leakage=ctx.leakage_surv,
        late_start_systemic_indices=systemic, late_start_local_indices=local,
        output_path=EXTENDED_DIR / "mean_surv_systemic_vs_local_clean_leakage.png", dpi=800)
    _, mae = plot_mean_surv_late_start(
        dm=ctx.dm, time_grid=ctx.time_grid, surv_clean=ctx.surv_oof, surv_leakage=ctx.leakage_surv,
        late_start_indices=late_start, output_path=EXTENDED_DIR / "mean_surv_late_start.png", dpi=800)
    rest = sorted(set(range(ctx.dm.shape[0])) - set(late_start))
    print_markdown_table(pd.DataFrame([summarize_group_rows(ctx.dm, rest, label="Rest"),
                                       summarize_group_rows(ctx.dm, late_start, label="Late start"),
                                       summarize_group_rows(ctx.dm, systemic, label="Late-start systemic"),
                                       summarize_group_rows(ctx.dm, local, label="Late-start local")]))
    print("\nCalibration error summaries:")
    print_markdown_table(pd.DataFrame([{"Curve comparison": key, "MAE": f"{value:.6f}"}
                                       for key, value in {**mae, **systemic_local_mae}.items()]))
    deltas = ctx.leakage_deltas
    print("\nGBSA leakage delta row:")
    print_markdown_table(deltas[deltas["model"] == ctx.model].reset_index(drop=True))


def emit_fig_ext_subgroup_time_perf(ctx: ManuscriptContext) -> None:
    model_stat = ctx.model_perf[ctx.model]
    for metric, y_limits, label, title in (("auc", (0.65, 0.95), "fig:auc-subgroup", "AUC"),
                                           ("ipcw", (0.61, 0.75), "fig:ipcw-subgroup", "IPCW-C")):
        subgroups = model_stat.hrher2_series.get(metric, {})
        plot_hrher2_time_curves(
            metric=metric, time_points=model_stat.series[metric].times, shade="std", shade_alpha=0.18,
            y_limits=y_limits, title=f"{DEFAULT_MODEL_NAME_MAP[ctx.model]} {title} by HR/HER2", figsize=(9.0, 6.0),
            dpi=800, output_path=PERFORMANCE_DIR / f"td_{metric}_subgroup.png", subgroup_series=subgroups)
        print(("\n" if metric == "ipcw" else "") + f"subfigure: {label}")
        series = {s: subgroups.get(s, subgroups.get(s.replace("/", "_"))) for s in SUBTYPE_ORDER}
        stats = horizon_summary({s: v for s, v in series.items() if v is not None})
        print_markdown_table(mean_sd(stats).rename(columns="{}d".format).rename_axis("Subtype").reset_index())


def emit_fig_grouped_shap(ctx: ManuscriptContext) -> None:
    plot_group_mean_abs_shap(
        shap_values=ctx.shap_oof, feature_names=ctx.features, group_map=ctx.group_map, label_map=ctx.feature_labels,
        top_k=10, title="Feature-group contribution share to 365d progression risk", figsize=(8.0, 5.0), dpi=800,
        output_path=SHAP_DIR / "shap_group_mean.png")
    mean_share = build_group_shap_share_df(ctx.shap_risk_oof, ctx.group_map).mean(axis=0)
    mean_share = mean_share.sort_values(ascending=False).head(10)
    print_markdown_table(pd.DataFrame({"Feature group": mean_share.index, "Mean normalized SHAP share": [
        f"{float(value):.4f}" for value in mean_share.to_numpy()]}))


def emit_fig_km_no_rad_prior(ctx: ManuscriptContext) -> None:
    dropped = ctx.no_rad_prior_df
    plot_km_modeled_vs_no_rad_prior(modeled_df=ctx.dm, no_rad_prior_df=dropped,
                                    output_path=EXTENDED_DIR / "km_modeled_vs_no_rad_prior.png", dpi=800)
    rows = []
    for cohort, df, time_col, event_col in (("Modeled cohort", ctx.dm, "PFS_TIME_DAYS", "PFS_EVENT"),
                                            ("Dropped: no radiology <=90d prior", dropped, "ORIGINAL_PFS_TIME_DAYS",
                                             "ORIGINAL_PFS_EVENT")):
        rows.append({"Cohort": cohort, "mLoTs": f"{df.shape[0]}", "Patients": f"{df['PATIENT_ID'].nunique()}",
                     "Event rate (%)": f"{float(np.mean(df[event_col] == 1) * 100):.2f}",
                     "Median PFS": format_median_ci(*median_pfs_ci(df[time_col], df[event_col]), digits=0)})
    print_markdown_table(pd.DataFrame(rows))


def emit_fig_continuous_calibration(ctx: ManuscriptContext) -> None:
    predictions = pd.read_csv("data/rebuttal/calibration/horizon_predictions.csv",
                              usecols=["cohort", "PATIENT_ID"])
    counts = (predictions[predictions["cohort"].isin(["DFCI", "VICC"])]
              .groupby("cohort").agg(retained_lines=("PATIENT_ID", "size"),
                                    retained_patients=("PATIENT_ID", "nunique")).reset_index())
    print("Drawn by analyses/calibration_figures.py from results/rebuttal/calibration/curves.csv. "
          "The calibration metrics are in Extended Data Table 25.")
    rows = [{"Cohort": "MSK-CHORD", "Lines": f"{ctx.dm.shape[0]}", "Patients": f"{ctx.dm['PATIENT_ID'].nunique()}"}]
    rows += [{"Cohort": row.cohort, "Lines": f"{row.retained_lines}", "Patients": f"{row.retained_patients}"}
             for row in counts.itertuples()]
    print_markdown_table(pd.DataFrame(rows))


def emit_fig_external_tertiles(ctx: ManuscriptContext) -> None:
    tertiles = pd.read_csv(RESULTS_PATH / "rebuttal" / "external" / "recalibration_tertiles.csv")
    tertiles["risk_tertile"] = pd.Categorical(tertiles["risk_tertile"], ["Low", "Middle", "High"])
    tertiles = tertiles.sort_values(["cohort", "horizon", "risk_tertile"])
    print("Drawn by analyses/calibration_figures.py. Tertiles of the original predicted 365-day risk.")
    percent = "{:.1f}".format
    print_markdown_table(pd.DataFrame({
        "Cohort": tertiles["cohort"], "Tertile": tertiles["risk_tertile"].astype(str),
        "Horizon (d)": tertiles["horizon"].astype(str), "Regimens": tertiles["n_lines"].astype(str),
        "Patients": tertiles["n_patients"].astype(str),
        "Observed KM (%)": (100 * tertiles["observed_KM_survival"]).map(percent),
        "Original predicted (%)": (100 * tertiles["predicted_survival"]).map(percent),
        "Local CV update (%)": (100 * tertiles["recalibrated_survival"]).map(percent),
        "At risk": tertiles["n_at_risk"].astype(str)}))
    event_free = tertiles[tertiles["horizon"] == 730].groupby("cohort")["n_at_risk"].sum()
    print("Regimens observed event-free through 730 days: "
          + ", ".join(f"{cohort} {count}" for cohort, count in event_free.items()))


def emit_text_numbers(ctx: ManuscriptContext) -> None:
    dm = ctx.dm
    censored = dm[EVENT_COL] == 0
    administrative = censored & (dm[TIME_COL] >= ADMIN_CENSOR_DAYS)
    eligible = pd.read_csv(RESULTS_PATH / "rebuttal" / "eligibility" / "late_start" / "line_classification.csv",
                           usecols=["RETAINED", "QC_CHEMO_ENDO", "QC_HRNEG", "LATE_START"])
    flagged, late = eligible["QC_CHEMO_ENDO"], eligible["LATE_START"]
    masking = genomic_masking(dm)
    ca15_3 = pd.read_csv(RESULTS_PATH / "rebuttal" / "external" / "six_variable_availability.csv",
                         index_col="cohort")["CA15_3_available_pct"]
    print_markdown_table(pd.DataFrame([
        ("PFS labeling and censoring", "Censored mLoTs", f"{censored.sum():,} of {len(dm):,}"),
        ("PFS labeling and censoring", "Censored after a documented non-progression assessment",
         f"{(censored & ~administrative).sum():,}"),
        ("PFS labeling and censoring", f"Administratively censored at {ADMIN_CENSOR_DAYS} d", f"{administrative.sum():,}"),
        ("PFS labeling and censoring", "QC exclusions among eligible lines",
         f"{(~eligible['RETAINED']).sum():,} of {len(eligible):,}"),
        ("PFS labeling and censoring", "Chemotherapy and endocrine therapy in the same mLoT", f"{flagged.sum():,}"),
        ("PFS labeling and censoring", "Endocrine therapy in HR-negative disease", f"{eligible['QC_HRNEG'].sum():,}"),
        ("PFS labeling and censoring", "Flagged lines with both classes in the first 28 days",
         f"{(flagged & ~late).sum():,}"),
        ("PFS labeling and censoring", "Flagged lines with the second class after day 28",
         f"{late.sum():,} ({100 * late.mean():.1f}% of eligible lines)"),
        ("Feature extraction", "mLoTs with genomic features masked",
         f"{masking['masked_lines']:,} of {masking['lines']:,}"),
        ("Feature extraction", "Patients with genomic features masked",
         f"{masking['masked_patients']:,} of {masking['patients']:,}"),
        ("External validation", "Regimens with CA15-3 recorded, DFCI", f"{ca15_3['DFCI']:.1f}%"),
        ("External validation", "Regimens with CA15-3 recorded, VICC", f"{ca15_3['VICC']:.1f}%"),
    ], columns=["Section", "Quantity", "Value"]))


MANUSCRIPT_DATA = [
    ("Figure 1", "Workflow overview", "fig:workflow, fig:cohort-construction",
     lambda ctx: print("Static BioRender figure; no data output.")),
    "t1",
    ("Figure 2", "Time-dependent model performance and population risk stratification", "fig:perf-and-risk",
     emit_fig_perf_and_risk),
    "t2",
    ("Figure 3", "Tertile-stratified Kaplan-Meier and mean predicted survival by subtype and line group",
     "fig:stratify-by-subtype", emit_fig_stratify_by_subtype),
    ("Figure 4", "Interpretability analysis for GBSA risk scores", "fig:shap-compact", emit_fig_shap_compact),
    ("Figure 5", "Modality ablations", "fig:ablation-only, fig:ablation-exclude", emit_fig_ablations),
    ("Main text", "Numbers quoted in the text and not shown in a table or figure", "sec:results, sec:methods",
     emit_text_numbers),
]
EXTENDED_DATA = [
    *(f"ed{i}" for i in range(1, 29)),
    ("Extended Data Fig. 1", "Subtype-stratified time-dependent discrimination of GBSA",
     "fig:ext-subgroup-time-perf", emit_fig_ext_subgroup_time_perf),
    ("Extended Data Fig. 2", "Mean normalized SHAP contributions aggregated by clinical feature groups",
     "fig:grouped-shap", emit_fig_grouped_shap),
    ("Extended Data Fig. 3", "Kaplan-Meier estimates of PFS for modeled mLoTs and lines excluded for missing "
     "radiology screening", "fig:km-no-rad-prior", emit_fig_km_no_rad_prior),
    ("Extended Data Fig. 4", "Late-start treatment leakage analysis", "fig:km-diff-preds", emit_fig_km_diff_preds),
    ("Extended Data Fig. 5", "Calibration curves at 180, 365 and 730 days", "fig:continuous-calibration",
     emit_fig_continuous_calibration),
    ("Extended Data Fig. 6", "Observed and predicted PFS by original predicted-risk tertile at DFCI and VICC",
     "fig:external-tertile-calibration", emit_fig_external_tertiles),
]
OUTPUTS = {"manuscript_data.txt": MANUSCRIPT_DATA, "extended_data.txt": EXTENDED_DATA}


def build_section(item, manuscript_ctx: ManuscriptContext, table_ctx: Context) -> str:
    if isinstance(item, str):
        with redirect_stdout(sys.stderr):
            module = load(item)
            panels = module.build(table_ctx)
        return render(module, panels)
    number, title, label, emit = item
    body = StringIO()
    with redirect_stdout(body):
        emit(manuscript_ctx)
    header = ["=" * WIDTH, f"{number}. {title}", f"[{label}]", "=" * WIDTH]
    return "\n".join(header) + "\n\n" + body.getvalue().strip("\n") + "\n"


def main() -> int:
    manuscript_ctx, table_ctx, failed, sections = ManuscriptContext(), Context(), [], {}
    items = [item for items in OUTPUTS.values() for item in items]
    for item in sorted(items, key=lambda item: isinstance(item, str)):
        name = item if isinstance(item, str) else item[0]
        print(f"Build {name}", file=sys.stderr, flush=True)
        try:
            sections[name] = build_section(item, manuscript_ctx, table_ctx)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            sections[name] = f"{name}: failed, see the error printed by manuscript.py\n"
    for path, items in OUTPUTS.items():
        Path(path).write_text("\n".join(sections[item if isinstance(item, str) else item[0]] for item in items))
        print(f"Wrote {path}", file=sys.stderr)
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
