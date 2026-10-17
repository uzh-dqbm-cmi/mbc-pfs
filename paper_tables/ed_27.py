import pandas as pd

from paper_tables.common import Panel
from analyses.cohort_selection.imaging_eligibility import CHANGE, imaging_eligibility

NUMBER = "Extended Data Table 27"
LABEL = "table:imaging-eligibility"
TITLE = "Sensitivity to the pre-line imaging requirement"

LABELS = {"Added 91-180-day group": "Added 91–180-day group", CHANGE: "Paired change, expanded minus original"}


def interval(point, lo, hi, sign=""):
    return f"{point:{sign}.4f} [{lo:.4f}, {hi:.4f}]"


def build(ctx):
    cohorts, change = imaging_eligibility(ctx.survival)
    rows = [{"Cohort": LABELS.get(r.Cohort, r.Cohort), "Lines": f"{r.Lines:,}", "Patients": f"{r.Patients:,}",
             "C (95% CI)": interval(r.C, r.CI_low, r.CI_high), "IBS": f"{r.IBS:.4f}"} for r in cohorts.itertuples()]
    rows += [{"Cohort": LABELS[r.Cohort], "Lines": "", "Patients": "", "C (95% CI)": interval(r.C, r.CI_low, r.CI_high, "+"),
              "IBS": ""} for r in change.itertuples()]
    return [Panel("", pd.DataFrame(rows))]
