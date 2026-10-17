import numpy as np
import pandas as pd
from tabulate import tabulate

from analyses.common.ibs_bootstrap import paired
from analyses.common.survival import SurvivalContext

REFERENCE = "ref_mlot1_ca15_3"
FULL = {"CoxPH": "full_coxph", "GBSA": "full_gbsa"}


def comparison(ctx):
    reference = ctx.evaluate(REFERENCE)
    rows = {}
    for label, name in FULL.items():
        diff = paired(ctx, name, REFERENCE)
        folds = np.array([ctx.evaluate(name)["fold_C"][fold] - value for fold, value in reference["fold_C"].items()])
        rows[label] = dict(diff, gain=diff.delta_C / (reference["C"] - .5), fold_low=folds.min(),
                           fold_high=folds.max(), favoring=(folds > 0).sum(), folds=len(folds))
    return ctx.pooled([REFERENCE, *FULL.values()]), pd.DataFrame.from_dict(rows, orient="index")


def generate(context=None):
    ctx = context or SurvivalContext()
    pooled, diffs = comparison(ctx)
    labels = {REFERENCE: "Simple clinical Cox reference", **{name: f"Full {label}" for label, name in FULL.items()}}
    rows = [[labels[r.Index], f"{r.C:.4f}", f"{r.C_low:.4f}–{r.C_high:.4f}", f"{r.IBS:.4f}"] for r in pooled.itertuples()]
    diffs = diffs.loc[["GBSA", "CoxPH"]]
    comparisons = [[f"Full {r.Index} minus reference", f"{r.delta_C:+.4f} ({r.CI_low:.4f}–{r.CI_high:.4f})",
                    f"{r.fold_low:.4f}–{r.fold_high:.4f}", f"{r.favoring} of {r.folds}", f"{r.gain:.2%}"]
                   for r in diffs.itertuples()]
    ibs_rows = [[f"Full {r.Index} minus reference", f"{r.delta_IBS:+.6f}", f"{r.IBS_low:+.6f} to {r.IBS_high:+.6f}"]
                for r in diffs.itertuples()]
    legacy = ctx.evaluate("clinical_cox")
    return "\n\n".join([
        "Simple clinical reference",
        f"Cohort: {len(ctx.dm):,} lines; {len(ctx.ids):,} patients.",
        "The reference uses IS_MLOT1, HR, HER2, latest ECOG, liver involvement in the last 90 days, and latest log1p CA15-3.",
        "Cox alpha = 1, Efron ties. No separate missing-value flags. Training-fold preprocessing fills missing numeric values.",
        "C and IBS use pooled out-of-fold predictions. Intervals use 1,000 paired patient draws, seed 42.",
        tabulate(rows, headers=["Model", "Pooled C", "95% patient-bootstrap CI", "IBS"], tablefmt="psql", disable_numparse=True),
        tabulate(comparisons, headers=["Comparison", "Change in C (paired 95% CI)", "Fold change range", "Folds favoring full model",
                                       "Relative gain above chance, change / (reference C - 0.5)"], tablefmt="psql", disable_numparse=True),
        "New IBS intervals: censoring KM and the observed censoring-time support are recalculated within each patient draw. Predictions stay fixed.",
        tabulate(ibs_rows, headers=["Comparison", "Change in pooled IBS", "95% paired patient-bootstrap CI"], tablefmt="psql", disable_numparse=True),
        f'Legacy author note: the older numeric-line reference with two missingness flags gives C = {legacy["C"]:.4f}. It is not the selected reference.'
    ]) + "\n"


if __name__ == "__main__":
    print(generate())
