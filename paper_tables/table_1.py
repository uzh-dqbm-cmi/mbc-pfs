import pandas as pd

from manuscript_utils import cohort_summary, top_treatment_combinations
from paper_tables.common import Panel

NUMBER = "Table 1"
LABEL = "table:cohort-summary"
TITLE = "Cohort size, PFS summary, and first-line (mLoT1) treatment-category combinations overall and by disease subtype"

FORMATS = {"Patients": "{:,}", "Total mLoTs": "{:,}", "Mean mLoTs/patient": "{:.2f}", "Event rate (%)": "{:.1f}",
           "Median PFS (days)": "{:.0f}", "First lines": "{:,}", "First-line event rate (%)": "{:.1f}"}
NAMES = {"HR-/HER2-": "TNBC"}
TOP_K = 2


def build(ctx):
    dm = ctx.manuscript.dm
    summary = cohort_summary(dm)
    table = pd.DataFrame({column: summary[column].map(fmt.format) for column, fmt in FORMATS.items()}).T
    rows = table.rename(columns=NAMES).rename_axis("").reset_index().to_dict("records")
    top = top_treatment_combinations(dm[dm.LINE == 1], "TREATMENT", TOP_K).replace({"subtype": NAMES})
    for rank, picks in top.groupby("rank"):
        picks = picks.set_index("subtype")
        rows.append({"": f"Top {rank} mLoT1", "Overall": "", **picks.regimen})
        rows.append({"": "– # (%)", "Overall": "", **{name: f"{row['count']:,} ({row.percentage:.1f}%)"
                                                     for name, row in picks.iterrows()}})
    return [Panel("a. Cohort characteristics", pd.DataFrame(rows))]
