import pandas as pd

from paper_tables.common import Panel
from analyses.site_count import MODELS, site_count

NUMBER = "Extended Data Table 26"
LABEL = "table:site-count"
TITLE = "Sensitivity to the number of recorded metastatic-site categories"


def build(ctx):
    pooled, diff = site_count(ctx.survival)
    pooled = pd.DataFrame([{"Model": MODELS[r.Index], "C (95% CI)": f"{r.C:.5f} [{r.C_low:.5f}, {r.C_high:.5f}]",
                            "IBS": f"{r.IBS:.5f}"} for r in pooled.itertuples()])
    paired = {"Comparison": "Site count minus original",
              "ΔC (paired 95% CI)": f"{diff.delta_C:+.5f} [{diff.CI_low:+.5f}, {diff.CI_high:+.5f}]",
              "ΔIBS (paired 95% CI)": f"{diff.delta_IBS:+.5f} [{diff.IBS_low:+.5f}, {diff.IBS_high:+.5f}]"}
    return [Panel("a. Pooled out-of-fold performance", pooled),
            Panel("b. Paired differences: site count minus original", pd.DataFrame([paired]))]
