import pandas as pd

from manuscript_utils import baseline_contrasts
from paper_tables.common import Panel

NUMBER = "Extended Data Table 5"
LABEL = "table:baseline-contrasts"
TITLE = "Feature-level baseline value contrasts between high- and low-risk tertiles (GBSA)"
COLUMN = "Top 5 baseline features (Cohen's d, H–L)"


def first_only(column):
    return column.where(~column.duplicated(), "")


def build(ctx):
    m = ctx.manuscript
    table = baseline_contrasts(m)
    table["label"] = [m.feature_labels.get(feature, feature) for feature in table.feature]
    top = table[table.kind == "top"]
    rows = pd.DataFrame({"Cohort": first_only(top.cohort),
                         COLUMN: [f"{label} ({d:+.3f})" for label, d in zip(top.label, top.d)]})
    return [Panel("", rows)]
