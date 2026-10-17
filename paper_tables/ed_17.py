from paper_tables.common import Panel
from analyses.external_validation import institution_counts, institution_metrics

NUMBER = "Extended Data Table 17"
LABEL = "table:genie-institution-metrics"
TITLE = "External evaluation on AACR Project GENIE BPC metastatic breast-cancer regimens at DFCI and VICC"

BLOCKS = {"pfs_i": "Released PFS-I: Progressing or death",
          "mixed": "PFS-I + Mixed (main analysis): Progressing or Mixed, or death"}


def build(ctx):
    table = institution_metrics()
    panels = [Panel(title, table.loc[table.endpoint.eq(endpoint)].drop(columns="endpoint"))
              for endpoint, title in BLOCKS.items()]
    counts = institution_counts()
    dropped = counts.loc["pfs_i", "n_lines"] - counts.loc["mixed", "n_lines"]
    panels[-1].note = (f"The PFS-I + Mixed endpoint excludes {dropped['DFCI']} additional DFCI and {dropped['VICC']} "
                       "additional VICC regimens with PFS below 28 days.")
    return panels
