from paper_tables.common import Panel, fold_performance

NUMBER = "Extended Data Table 8"
LABEL = "table:leakage-performance"
TITLE = "Global performance metrics without late-start treatment masking"


def build(ctx):
    return [Panel("", fold_performance(ctx.manuscript.leakage_perf))]
