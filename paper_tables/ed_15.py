import pandas as pd

from paper_tables.common import Panel

NUMBER = "Extended Data Table 15"
LABEL = "table:qc-exclusions"
TITLE = "Sensitivity to the QC exclusions"


def build(ctx):
    table = pd.read_csv("results/rebuttal/eligibility/qc_sensitivity.csv")
    table["Lines"] = table.Lines.map("{:,}".format)
    table["Patients"] = table.Patients.map("{:,}".format)
    table["C"] = table.C.map("{:.4f}".format)
    table["IBS"] = table.IBS.map("{:.4f}".format)
    return [Panel("", table)]
