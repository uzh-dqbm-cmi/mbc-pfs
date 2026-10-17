import pandas as pd

from manuscript_utils import group_contrasts
from paper_tables.common import Panel
from paper_tables.ed_05 import first_only

NUMBER = "Extended Data Table 6"
LABEL = "table:shap-group-contrasts"
TITLE = "Group-level SHAP contribution contrasts between high- and low-risk tertiles (GBSA)"


def build(ctx):
    table = group_contrasts(ctx.manuscript)
    top = table[table.kind == "top"]
    rows = pd.DataFrame({"Cohort": first_only(top.cohort), "Top 5 feature groups": top.feature,
                         "Cohen's d (H–L)": top.d.map("{:+.3f}".format)})
    return [Panel("", rows)]
