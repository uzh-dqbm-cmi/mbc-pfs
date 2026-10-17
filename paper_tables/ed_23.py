import pandas as pd

from paper_tables.common import Panel
from analyses.clinical_comparator import FULL, REFERENCE, comparison

NUMBER = "Extended Data Table 23"
LABEL = "table:clinical-comparator"
TITLE = "Clinical Cox reference and full multimodal models"

MODELS = {REFERENCE: "Six-variable clinical Cox", **{name: f"Full {label}" for label, name in FULL.items()}}


def build(ctx):
    pooled, diffs = comparison(ctx.survival)
    pooled = pd.DataFrame([(MODELS[r.Index], f"{r.C:.4f} [{r.C_low:.4f}, {r.C_high:.4f}]", f"{r.IBS:.4f}")
                           for r in pooled.itertuples()], columns=["Model", "C (95% CI)", "IBS"])
    paired = pd.DataFrame([(r.Index, f"{r.delta_C:+.4f} [{r.CI_low:.4f}, {r.CI_high:.4f}]",
                            f"{r.delta_IBS:+.6f} [{r.IBS_low:+.6f}, {r.IBS_high:+.6f}]", f"{r.gain:.0%}",
                            f"{r.fold_low:.4f} to {r.fold_high:.4f}", f"{r.favoring} of {r.folds}")
                           for r in diffs.itertuples()],
                          columns=["Full model", "ΔC (paired 95% CI)", "ΔIBS (paired 95% CI)",
                                   "Relative gain above chance", "Within-fold ΔC", "Folds favoring full model"])
    return [Panel("a. Pooled out-of-fold performance", pooled),
            Panel("b. Paired differences: full model minus clinical reference", paired)]
