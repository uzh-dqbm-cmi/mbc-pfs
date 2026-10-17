import pandas as pd

from paper_tables.common import Panel
from analyses.care_process import FULL, REDUCED, line_groups, reduced_model

NUMBER = "Extended Data Table 24"
LABEL = "table:line-status-missingness"
TITLE = "Sensitivity to first-line status and missingness indicators, and discrimination within line groups"

MODELS = {FULL: "Full GBSA", REDUCED: "GBSA without first-line status and missingness indicators"}
GROUPS = {"all": "All lines", "first": "First lines (mLoT1)", "later": "Later lines (mLoT2+)"}


def with_ci(r):
    return f"{r.C:.4f} [{r.C_low:.4f}, {r.C_high:.4f}]"


def build(ctx):
    pooled, diff = reduced_model(ctx.survival)
    pooled = pd.DataFrame([(MODELS[r.Index], with_ci(r), f"{r.IBS:.4f}") for r in pooled.itertuples()],
                          columns=["Model", "C (95% CI)", "IBS"])
    paired = pd.DataFrame([("Reduced minus full GBSA",
                            f"{diff.delta_C:+.6f} [{diff.CI_low:+.6f}, {diff.CI_high:+.6f}]",
                            f"{diff.delta_IBS:+.6f} [{diff.IBS_low:+.6f}, {diff.IBS_high:+.6f}]")],
                          columns=["Comparison", "ΔC (paired 95% CI)", "ΔIBS (paired 95% CI)"])
    groups = pd.DataFrame([(GROUPS[r.Index], f"{r.lines:,} ({r.patients:,})", with_ci(r))
                           for r in line_groups(ctx.survival).itertuples()],
                          columns=["Line group", "Lines (patients)", "C (95% CI)"])
    return [Panel("a. Pooled out-of-fold performance", pooled),
            Panel("b. Paired difference: reduced minus full GBSA", paired),
            Panel("c. Full GBSA within line groups", groups)]
