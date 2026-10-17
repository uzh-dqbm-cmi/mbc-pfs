from dataclasses import dataclass
from functools import cached_property

import pandas as pd
from tabulate import tabulate

WIDTH = 100


@dataclass
class Panel:
    title: str
    table: pd.DataFrame


class Context:
    def __init__(self):
        self.cache = {}

    @cached_property
    def manuscript(self):
        from manuscript_utils import ManuscriptContext
        return ManuscriptContext()

    @cached_property
    def survival(self):
        from analyses.common.survival import SurvivalContext
        return SurvivalContext()


def fold_performance(model_perf):
    from manuscript_utils import DEFAULT_MODEL_NAME_MAP, mean_sd, overall_performance
    cells = mean_sd(overall_performance(model_perf)).rename(columns=DEFAULT_MODEL_NAME_MAP)
    return cells.rename_axis("Metric").reset_index()


def render_panel(panel):
    parts = [panel.title] if panel.title else []
    table = panel.table.astype(str)
    parts.append(tabulate(table, headers="keys", showindex=False, tablefmt="psql", disable_numparse=True))
    return "\n".join(parts)


def render(module, panels):
    header = ["=" * WIDTH, f"{module.NUMBER}. {module.TITLE}", "=" * WIDTH]
    return "\n".join(header) + "\n\n" + "\n\n".join(render_panel(panel) for panel in panels) + "\n"
