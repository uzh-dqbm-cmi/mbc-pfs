from manuscript_utils import DEFAULT_MODEL_NAME_MAP, horizon_summary, mean_sd
from paper_tables.common import Panel

NUMBER = "Extended Data Table 4"
LABEL = "table:all-models-time-metrics"
TITLE = "Per-horizon performance across models for PFS prediction"

PANELS = {"a. IPCW-C": "ipcw", "b. AUC": "auc", "c. Brier score": "brier_score"}


def table(model_perf, key):
    cells = mean_sd(horizon_summary({DEFAULT_MODEL_NAME_MAP[model]: stat.series[key] for model, stat in model_perf.items()}))
    return cells.rename(columns="{} d".format).rename_axis("Model").reset_index()


def build(ctx):
    return [Panel(title, table(ctx.manuscript.model_perf, key)) for title, key in PANELS.items()]
