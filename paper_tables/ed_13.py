from paper_tables.common import Panel
from analyses.common.primary_bootstrap import summary

NUMBER = "Extended Data Table 13"
LABEL = "table:overall-performance-bootstrap-ci"
TITLE = "Global performance metrics from pooled out-of-fold predictions for all modeled mLoTs"


def build(ctx):
    return [Panel("", summary())]
