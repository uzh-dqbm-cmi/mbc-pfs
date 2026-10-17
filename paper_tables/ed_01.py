import pandas as pd

from manuscript_utils import age_table, baseline_characteristics
from paper_tables.common import Panel

NUMBER = "Extended Data Table 1"
LABEL = "table:extended-data-tables"
TITLE = "Extended Data tables for the breast cancer subset"


def percent(numerator, denominator):
    # Percent within age bin
    return (100 * numerator / denominator).map("{:.1f}".format)


def baseline(dm):
    b = baseline_characteristics(dm)
    rows = [("n", f"{b['patients']:,}"),
            ("Age at first modeled line initiation, mean (sd)", f"{b['age_mean']:.1f} ({b['age_sd']:.1f})"),
            ("Stage at diagnosis", ""),
            ("Stage I–III", f"{b['patients'] - b['stage_iv']:,} ({(1 - b['stage_iv_share']) * 100:.1f}%)"),
            ("Stage IV", f"{b['stage_iv']:,} ({b['stage_iv_share'] * 100:.1f}%)")]
    return pd.DataFrame(rows, columns=["Variable", "Overall"])


def by_age(dm):
    t = age_table(dm, True)
    n = t.Patients
    table = pd.DataFrame({"Age group": t["Age group"].astype(str).str.replace("<=", "≤").str.replace("-", "–"),
                          "HR+": percent(t["HR positive"], n), "HR-": percent(n - t["HR positive"], n),
                          "HER2+": percent(t["HER2 positive"], n), "HER2-": percent(n - t["HER2 positive"], n)})
    return table


def build(ctx):
    dm = ctx.manuscript.dm
    return [Panel("a. Baseline characteristics of the breast cancer subset", baseline(dm)),
            Panel("b. Patient-level HR/HER2 status by age at the first retained modeled line (row-wise %)", by_age(dm))]
