import pandas as pd

from manuscript_utils import top_treatment_combinations
from paper_tables.common import Panel

NUMBER = "Extended Data Table 2"
LABEL = "table:subtype-top-tx-lot1"
TITLE = "Most common first-line metastatic drug combinations at line initiation (mLoT1), stratified by disease subtypes"
TOP_K = 5


def combinations(dm, col_name):
    top = top_treatment_combinations(dm, col_name, TOP_K)
    return pd.DataFrame({"Subtype": top.subtype.where(top["rank"] == 1, ""), "Rank": top["rank"].astype(str),
                         "Regimen": top.regimen,
                         "# (%)": [f"{n:,} ({pct:.1f}%)" for n, pct in zip(top["count"], top.percentage)]})


def build(ctx):
    dm = ctx.manuscript.dm
    return [Panel("", combinations(dm[dm.LINE == 1], "AGENT"),
                  "Percentages reflect the share within each HR/HER2 subtype.")]
