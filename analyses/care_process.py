import numpy as np
import pandas as pd

from analyses.common.ibs_bootstrap import paired
from analyses.common.survival import ci, patient_bootstrap

FULL, REDUCED = "full_gbsa", "gbsa_no_line_missingness"


def reduced_model(ctx):
    return ctx.pooled([FULL, REDUCED]), paired(ctx, REDUCED, FULL)


def line_groups(ctx):
    dm, full = ctx.dm, ctx.evaluate(FULL)
    rows = []
    for group, mask in [("all", np.ones(len(dm), dtype=bool)), ("first", dm.LINE.eq(1).to_numpy()),
                        ("later", dm.LINE.gt(1).to_numpy())]:
        lines = dm.loc[mask]
        result = full if mask.all() else patient_bootstrap(full["sf"].to_numpy()[mask], full["grid"], lines)
        low, high = ci(result["samples"])
        rows.append(dict(group=group, lines=len(lines), patients=lines.PATIENT_ID.nunique(), C=result["C"],
                         C_low=low, C_high=high, decrease=full["C"] - result["C"]))
    return pd.DataFrame(rows).set_index("group")
