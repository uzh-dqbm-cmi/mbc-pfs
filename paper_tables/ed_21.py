import pandas as pd

from paper_tables.common import Panel
from analyses.common import cohort as C
from analyses.reconstruction import km_ci
from analyses.progression_definition import WINDOWS, brier, concordance, sensitivity

NUMBER = "Extended Data Table 21"
LABEL = "table:report-density"
TITLE = "First-line and all-line endpoint sensitivities in MSK-CHORD"

RULES = {"frozen": "Original (any progression report)",
         **{f"no_negative_{w}": f"No non-progression report within {w} days" for w in WINDOWS}}
GROUPS = {True: "First captured lines (mLoT1)", False: "All mLoTs"}
COLUMNS = ["Progression definition", "n", "Events", "Median PFS, days (95% CI)", "Time-dependent C",
           "Brier score at 365 days"]


def median(r):
    return km_ci(r, "–")


def build(ctx):
    dm = C.load_design_matrix()
    results = {first: sensitivity(dm, first_only=first).set_index("rule") for first in GROUPS}
    rows = []
    for first, group in GROUPS.items():
        rows.append([group] + [""] * (len(COLUMNS) - 1))
        for rule, label in RULES.items():
            r = results[first].loc[rule]
            rows.append([f"– {label}", f"{int(r.n):,}", f"{int(r.events):,}", median(r), concordance(r), brier(r)])
    first = results[True]
    original, midpoint = first.loc["frozen"], first.loc["midpoint"]
    continued = pd.DataFrame(
        [[label, f(original), f(midpoint)] for label, f in
         [("Median PFS, days (95% CI)", median), ("Time-dependent C, mean ± fold SD", concordance),
          ("Brier score at 365 days, mean ± fold SD", brier)]],
        columns=["Measure", "Original scan date", "Midpoint imputation"])
    return [Panel("", pd.DataFrame(rows, columns=COLUMNS)),
            Panel(f"(continued) Midpoint-imputation sensitivity in {int(original.n):,} first captured metastatic lines",
                  continued)]
