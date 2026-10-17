from paper_tables.common import Panel
from paper_tables.ed_02 import combinations

NUMBER = "Extended Data Table 3"
LABEL = "table:subtype-top-tx-lot2plus"
TITLE = "Most common mLoT2+ treatment-category combinations at mLoT initiation, stratified by disease subtypes"


def build(ctx):
    dm = ctx.manuscript.dm
    return [Panel("", combinations(dm[dm.LINE != 1], "TREATMENT"))]
