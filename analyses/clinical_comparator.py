import numpy as np
import pandas as pd

from analyses.common.ibs_bootstrap import paired

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
