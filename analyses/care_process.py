import numpy as np
import pandas as pd
from tabulate import tabulate

from analyses.common.ibs_bootstrap import paired
from analyses.common.survival import SurvivalContext, ci, line_missingness_columns, patient_bootstrap

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


def generate(context=None):
    ctx = context or SurvivalContext()
    dm = ctx.dm
    pooled, diff = reduced_model(ctx)
    full, reduced = pooled.loc[FULL], pooled.loc[REDUCED]
    removed = line_missingness_columns(dm)
    labels = {"all": "All lines", "first": "First captured line (mLoT1)", "later": "Later captured lines (mLoT2+)"}
    rows = [[labels[r.Index], f"{r.lines:,}", f"{r.patients:,}", f"{r.C:.4f}", f"{r.C_low:.4f}–{r.C_high:.4f}",
             f"{r.decrease:.4f}"] for r in line_groups(ctx).itertuples()]
    return "\n\n".join([
        "First-line status and missingness",
        f"Cohort: {len(dm):,} lines; {len(ctx.ids):,} patients.",
        f"Removed inputs: {len(removed)} = IS_MLOT1 plus {len(removed)-1} explicit missingness or unknown flags.",
        ", ".join(removed),
        "C and IBS use pooled out-of-fold predictions. Change = reduced model minus full GBSA.",
        tabulate([["C", f"{full.C:.6f}", f"{reduced.C:.6f} [{reduced.C_low:.4f}, {reduced.C_high:.4f}]",
                   f"{diff.delta_C:+.6f}", f"{diff.CI_low:.6f} to {diff.CI_high:.6f}"],
                  ["IBS", f"{full.IBS:.6f}", f"{reduced.IBS:.6f}", f"{diff.delta_IBS:+.6f}",
                   f"{diff.IBS_low:+.6f} to {diff.IBS_high:+.6f}"]],
                 headers=["Metric", "Full GBSA", f"{len(removed)} inputs removed", "Change", "Paired 95% CI"], tablefmt="psql", disable_numparse=True),
        "New IBS interval: 1,000 shared patient draws, seed 42. Censoring KM and the observed censoring-time support are recalculated within each draw.",
        tabulate(rows, headers=["Group", "Lines", "Patients", "C", "95% patient-bootstrap CI", "Decrease from all-line C"], tablefmt="psql", disable_numparse=True),
        "Each group uses its own 1,000 patient draws with seed 42. Predictions stay fixed.",
        "The treatment-modality ablation removes IS_MLOT1 with the treatment inputs. The surveillance-pattern ablation keeps IS_MLOT1 and unknown flags.",
        "Other inputs can retain care-process information. These results do not establish independence from the care process."
    ]) + "\n"


if __name__ == "__main__":
    print(generate())
