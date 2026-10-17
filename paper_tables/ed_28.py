from paper_tables.common import Panel
from analyses.cohort_selection import extended_data_table as E

NUMBER = "Extended Data Table 28"
LABEL = "table:qc-late-start"
TITLE = "Late-start chemotherapy plus endocrine QC exclusions in MSK-CHORD"

NOTE = ("Slope is the IPCW logistic calibration slope (ideal 1); ICI is in percentage points. Predicted - observed is computed "
        "before rounding; negative values indicate under-prediction of PFS.")


def build(ctx):
    b, c = E.panel_b(), E.panel_c()
    original, cohort = b.Lines.iat[0], c.loc[c.Stratum.eq("All lines"), "Lines"].iat[0]
    return [
        Panel("a. Lines flagged by the chemotherapy plus endocrine rule", E.panel_a()),
        Panel(f"b. Refits: discrimination on the original {original} lines", b),
        Panel(f"c. One-year calibration in the {cohort}-line cohort, original and refit models", c, NOTE),
        Panel("d. External evaluation before recalibration, original / refit", E.panel_d()),
        Panel("e. Reconstruction agreement with GENIE BPC before QC", E.panel_e()),
        Panel("f. Late-start BPC episodes excluded by the chemotherapy plus endocrine rule", E.panel_f()),
        Panel("g. First-line median PFS with late-start lines restored", E.panel_g())]
