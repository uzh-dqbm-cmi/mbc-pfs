from paper_tables.common import Panel, fold_performance

NUMBER = "Table 2"
LABEL = "table:overall-performance"
TITLE = "Global performance across survival models for PFS prediction"


def build(ctx):
    return [Panel("", fold_performance(ctx.manuscript.model_perf))]
