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

from manuscript_utils import (
    ManuscriptContext,
    emit_fig_ablation_exclude,
    emit_fig_ablation_only,
    emit_fig_ext_subgroup_time_perf,
    emit_fig_grouped_shap,
    emit_fig_km_diff_preds,
    emit_fig_km_no_rad_prior,
    emit_fig_perf_and_risk,
    emit_fig_shap_compact,
    emit_fig_stratify_by_subtype,
    emit_placeholder,
    emit_table_age_ablation_deltas,
    emit_table_age_ablation_deltas_less_regularized,
    # emit_table_age_ablation_deltas_less_regularized,
    emit_table_age_ablation_performance,
    emit_table_all_models_time_metrics,
    emit_table_baseline,
    emit_table_baseline_contrasts,
    emit_table_cohort_summary,
    emit_table_feature_groups,
    emit_table_hr_her2_by_age,
    emit_table_leakage_deltas,
    emit_table_leakage_performance,
    emit_table_overall_performance,
    emit_table_overall_performance_bootstrap_ci,
    emit_table_pfs_by_age_bin,
    emit_table_selected_model_configs,
    emit_table_shap_group_contrasts,
    emit_table_subtype_top_tx_lot1,
    emit_table_subtype_top_tx_lot2plus,
)


# %%
ctx = ManuscriptContext()


# %%
print("fig:workflow")
emit_placeholder("Static BioRender workflow figure; manuscript.py has no programmatic data output for it.")

print()
print("fig:cohort-construction")
emit_placeholder("Static BioRender cohort-construction figure; manuscript.py has no programmatic data output for it.")

print()
print("table:cohort-summary")
emit_table_cohort_summary(ctx)

print()
print("fig:perf-and-risk")
emit_fig_perf_and_risk(ctx)

print()
print("table:overall-performance")
emit_table_overall_performance(ctx)

print()
print("fig:stratify-by-subtype")
emit_fig_stratify_by_subtype(ctx)

print()
print("fig:shap-compact")
emit_fig_shap_compact(ctx)

print()
print("fig:ablation-only")
emit_fig_ablation_only(ctx)

print()
print("fig:ablation-exclude")
emit_fig_ablation_exclude(ctx)

print()
print("fig:km-diff-preds")
emit_fig_km_diff_preds(ctx)

print()
print("table:extended-data-tables")
emit_placeholder("Container label for Extended Data Tables 1A-1C; see the three labeled subtables that follow.")

print()
print("table:baseline")
emit_table_baseline(ctx)

print()
print("table:hr-her2-by-age")
emit_table_hr_her2_by_age(ctx)

print()
print("table:pfs-by-age-bin")
emit_table_pfs_by_age_bin(ctx)

print()
print("table:subtype-top-tx-lot1")
emit_table_subtype_top_tx_lot1(ctx)

print()
print("table:subtype-top-tx-lot2plus")
emit_table_subtype_top_tx_lot2plus(ctx)

print()
print("table:all-models-time-metrics")
emit_table_all_models_time_metrics(ctx)

print()
print("fig:ext-subgroup-time-perf")
emit_fig_ext_subgroup_time_perf(ctx)

print()
print("table:baseline-contrasts")
emit_table_baseline_contrasts(ctx)

print()
print("table:shap-group-contrasts")
emit_table_shap_group_contrasts(ctx)

print()
print("fig:grouped-shap")
emit_fig_grouped_shap(ctx)

print()
print("table:leakage-deltas")
emit_table_leakage_deltas(ctx)

print()
print("table:leakage-performance")
emit_table_leakage_performance(ctx)

print()
print("table:age-ablation-deltas")
emit_table_age_ablation_deltas(ctx)

print()
print("table:age-ablation-deltas-less-regularized")
emit_table_age_ablation_deltas_less_regularized(ctx)

print()
print("table:age-ablation-performance")
emit_table_age_ablation_performance(ctx)

print()
print("fig:km-no-rad-prior")
emit_fig_km_no_rad_prior(ctx)

print()
print("table:selected-model-configs")
emit_table_selected_model_configs(ctx)

print()
print("table:feature-groups")
emit_table_feature_groups(ctx)

print()
print("table:overall-performance-bootstrap-ci")
emit_table_overall_performance_bootstrap_ci(ctx)
