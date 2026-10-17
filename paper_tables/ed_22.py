import pandas as pd

from paper_tables.common import Panel
from analyses.repeated_lines import (SELECTIONS, leave_one_out, one_line_per_patient,
                                                               pair_composition, paired_test)

NUMBER = "Extended Data Table 22"
LABEL = "table:repeated-lines"
TITLE = "Repeated treatment lines"

PAIRS = {"all": "All comparable pairs (original pooled C)", "within": "Within-patient pairs only",
         "between": "Between-patient pairs only"}


def build(ctx):
    survival = ctx.survival
    gbsa = survival.evaluate("full_gbsa")
    pairs = pd.DataFrame([(PAIRS[r.Index], f"{int(r.pairs):,}", f"{r.C:.4f}") for r in pair_composition(gbsa).itertuples()],
                         columns=["Pairs included", "Comparable ordered pairs", "C"])
    single = one_line_per_patient(survival, gbsa)
    selections = pd.DataFrame(
        [("All modeled lines (original pooled C)", f"{len(survival.dm):,}", f"{gbsa['C']:.4f}"),
         (f"One random line per patient, mean (SD) of {SELECTIONS} selections", f"{single['patients']:,}",
          f"{single['mean']:.4f} ({single['sd']:.4f})")],
        columns=["Evaluation set", "Lines", "C"])
    _, delta, low, high, _ = paired_test("", single["mean"] - gbsa["C"],
                                         single["leave_out"] - leave_one_out(gbsa["num"], gbsa["den"]))
    paired = pd.DataFrame([{"Comparison": "Mean one-line minus pooled C", "ΔC": f"{delta:+.5f}",
                            "95% patient-jackknife CI": f"{low:+.5f} to {high:+.5f}"}])
    return [Panel("", pairs), Panel("(continued) One randomly selected line per patient", selections),
            Panel("(continued) Paired difference", paired)]
