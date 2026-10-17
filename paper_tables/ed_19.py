import pandas as pd

from paper_tables.common import Panel
from analyses import reconstruction as M

NUMBER = "Extended Data Table 19"
LABEL = "table:bpc-reconstruction"
TITLE = "Agreement of the mLoT reconstruction with curated GENIE BPC regimens at DFCI, VICC and MSK, using BPC PFS-I + Mixed (Progressing or Mixed imaging, or death)"


def conditional_agreement(d):
    return pd.DataFrame([{"BPC site": r.SITE, "Same regimen / all retained": M.frac(r.PAIRS, r.RETAINED),
                          "Same event indicator, given same regimen": M.frac(r.SAME_LABEL, r.PAIRS),
                          "Same event indicator and exact PFS duration, given same regimen": M.frac(r.EXACT, r.PAIRS),
                          "Cohen's kappa": f"{r.KAPPA:.3f}"}
                         for r in M.conditional_agreement(d.lines, d.pairs).itertuples()])


def regimen_categories(d):
    r = M.regimen_category(d.lines, d.courses, d.regimens, d.ref)
    return M.breakdown(r.CATEGORY, r.SITE, M.REGIMEN_ROWS, "episodes",
                       ("Same regimen or a listed grouping difference (rows 1–3)", M.REGIMEN_ROWS[:3]))


def endpoint_categories(d):
    return M.breakdown(M.endpoint_category(d.pairs), d.pairs.SITE, M.ENDPOINT_ROWS, "pairs",
                       ("Exact agreement or a listed endpoint-rule difference (rows 1–4)", M.ENDPOINT_ROWS[:4]))


def build(ctx):
    d = M.reconstruction()
    return [Panel("a. Patient and episode flow", M.retained_flow(d.lines, d.g, d.ref)),
            Panel("b. Agreement with conditional denominators", conditional_agreement(d)),
            Panel("c. Regimen disagreement categories", regimen_categories(d)),
            Panel("d. Endpoint disagreement categories", endpoint_categories(d))]
