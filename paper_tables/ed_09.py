from paper_tables.common import Panel
from paper_tables.ed_07 import deltas

NUMBER = "Extended Data Table 9"
LABEL = "table:age-ablation-deltas"
TITLE = "Descriptive mean performance differences across five outer test folds, with age excluded versus included"


def build(ctx):
    return [Panel("", deltas(ctx.manuscript.age_ablation_deltas))]
