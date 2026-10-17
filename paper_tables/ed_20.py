import pandas as pd

from paper_tables.common import Panel
from analyses.reconstruction import km_ci
from analyses.regimen_pfs import DEMICHELE, LITERATURE, regimen_medians

NUMBER = "Extended Data Table 20"
LABEL = "table:regimen-pfs"
TITLE = "First-line median PFS by exact regimen in MSK-CHORD and GENIE BPC, with a published benchmark"

SUBTYPE_LABELS = {"TNBC": "HR-/HER2-"}


def cells(source, k):
    return {f"{source} n": f"{k['n']:,}", f"{source} Events": f"{k['events']:,}",
            f"{source} Median (95% CI)": km_ci(k, "–")}


def medians():
    rows = []
    for r in regimen_medians().itertuples():
        label = SUBTYPE_LABELS.get(r.subtype, r.subtype)
        if not rows or rows[-1]["group"] != label:
            rows.append({"group": label, "Subtype and exact drug set": label})
        rows.append({"group": label, "Subtype and exact drug set": r.drugs.lower().capitalize(),
                     **cells("MSK-CHORD", r.msk), **cells("GENIE BPC PFS-I + Mixed", r.bpc)})
    return pd.DataFrame(rows).drop(columns="group").fillna("")


def benchmark():
    return pd.DataFrame([{"Treatment": v["arm"], "Patients (unweighted)": f"{v['n']:,}",
                          "Median PFS, months (95% CI)": f"{v['median']:.1f} ({v['lo']:.1f}–{v['hi']:.1f})"}
                         for v in LITERATURE.values() if v["source"] == DEMICHELE])


def build(ctx):
    return [Panel("a. Regimen medians", medians()),
            Panel("b. Published real-world benchmark (DeMichele et al.)", benchmark())]
