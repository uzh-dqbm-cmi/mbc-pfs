from paper_tables.common import Panel

NUMBER = "Extended Data Table 7"
LABEL = "table:leakage-deltas"
TITLE = "Descriptive mean performance differences across five outer test folds, without versus with treatment masking"

COLUMNS = {"delta_C": "ΔC", "delta_IBS": "Δ IBS", "delta_AUC": "Δ AUC"}


def deltas(table):
    table = table[["model", *COLUMNS]].copy()
    for column in COLUMNS:
        table[column] = table[column].map("{:.6f}".format)
    return table.rename(columns=COLUMNS)


def build(ctx):
    return [Panel("", deltas(ctx.manuscript.leakage_deltas))]
