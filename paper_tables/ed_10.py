from paper_tables.common import Panel, fold_performance

NUMBER = "Extended Data Table 10"
LABEL = "table:age-ablation-performance"
TITLE = "Global performance metrics with age excluded from baseline covariates (age ablation)"


def build(ctx):
    return [Panel("", fold_performance(ctx.manuscript.age_ablation_perf))]
