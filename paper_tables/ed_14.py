import pandas as pd

from manuscript_utils import DEFAULT_MODEL_NAME_MAP, selected_model_configs
from paper_tables.common import Panel

NUMBER = "Extended Data Table 14"
LABEL = "table:selected-model-configs"
TITLE = "Best-performing model configurations selected on outer folds, with selection counts"


def build(ctx):
    configs = selected_model_configs(ctx.manuscript.model_results_paths)
    model = configs.model.map(DEFAULT_MODEL_NAME_MAP)
    return [Panel("", pd.DataFrame({"Model": model.where(~model.duplicated(), ""), "Selected configurations (count)": [
        f"{config} ({count})" for config, count in zip(configs.config, configs["count"])]}))]
