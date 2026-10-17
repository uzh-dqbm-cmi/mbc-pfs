import pandas as pd

from paper_tables.common import Panel
from analyses.calibration import HORIZONS, OUT, calibration_table, cell
from analyses.pooled_calibration import OUTPUT as POOLED

NUMBER = "Extended Data Table 25"
LABEL = "table:calibration"
TITLE = "Internal and external calibration at 180, 365 and 730 days"

UPDATED = "intercept and slope, 5-fold CV"
INTERNAL = {"citl": "CITL", "slope": "Calibration slope", "ICI": "ICI (percentage points)",
            "E50": "E50 (percentage points)", "E90": "E90 (percentage points)"}
UPDATE = {"ICI": "ICI", "citl": "CITL", "slope": "Calibration slope", "E50": "E50", "E90": "E90"}
COHORTS = {"MSK_OOF": ("a", "MSK-CHORD, out-of-fold predictions", "lines"),
           "DFCI": ("b", "DFCI, BPC PFS-I + Mixed", "regimens"),
           "VICC": ("c", "VICC, BPC PFS-I + Mixed", "regimens")}


def days(h):
    return f"{h} days"


def internal_panels(metrics):
    panels = []
    for cohort, (letter, name, unit) in COHORTS.items():
        first = metrics.loc[metrics.cohort.eq(cohort)].iloc[0]
        panels.append(Panel(f"{letter}. {name} ({first.n_lines:,} {unit}, {first.n_patients:,} patients)",
                            calibration_table(metrics, cohort, INTERNAL)))
    return panels


def update_panels(recalibration):
    panels = []
    for letter, site in (("d", "DFCI"), ("e", "VICC")):
        d = recalibration.loc[recalibration.cohort.eq(site)].set_index(["variant", "horizon"])
        rows = []
        for key, label in UPDATE.items():
            for variant, name in (("original", "Original"), (UPDATED, "Updated")):
                rows.append({"Measure": label if variant == "original" else "", "Predictions": name,
                             **{days(h): cell(d.loc[(variant, h)], key) for h in HORIZONS}})
        panels.append(Panel(f"{letter}. {site}: before and after recalibration", pd.DataFrame(rows)))
    return panels


def pooled_panel(pooled):
    d = pooled.set_index(["variant", "horizon_days"])
    rows = [{"Measure": "ICI" if name == "Original" else "", "Predictions": name,
             **{days(h): f"{d.loc[(variant, h), 'ICI_percentage_points']:.2f}" for h in HORIZONS}}
            for variant, name in (("Original", "Original"), ("Site-specific CV update", "Updated"))]
    first = pooled.iloc[0]
    return Panel(f"f. DFCI and VICC pooled ({first.lines:,} regimens, {first.patients:,} patients): "
                 "ICI before and after the site-specific updates", pd.DataFrame(rows))


def build(ctx):
    metrics = pd.read_csv(OUT / "metrics.csv")
    recalibration = pd.read_csv("results/rebuttal/external/recalibration_metrics.csv")
    pooled = pd.read_csv(POOLED)
    return internal_panels(metrics) + update_panels(recalibration) + [pooled_panel(pooled)]
