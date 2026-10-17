from paper_tables.common import Panel
from paper_tables.ed_11 import window_performance

NUMBER = "Extended Data Table 12"
LABEL = "table:sensitivity-42d-performance"
TITLE = "Global performance metrics under the 42-day treatment-window sensitivity analysis"

DAYS = 42


def build(ctx):
    return [Panel("", window_performance(DAYS))]
