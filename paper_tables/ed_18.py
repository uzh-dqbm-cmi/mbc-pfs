import pandas as pd

from manuscript_utils import inventory
from paper_tables.common import Panel

NUMBER = "Extended Data Table 18"
LABEL = "table:model-inputs"
TITLE = "Complete model input list"


def build(ctx):
    features = inventory()
    panels = []
    for modality, part in features.groupby("modality", sort=False):
        table = pd.DataFrame({"Input name": part.feature, "Feature group": part.feature_group,
                              "SP": part.surveillance_pattern.map({True: "Yes", False: "No"})})
        panels.append(Panel(f"{modality} ({len(part)} inputs)", table))
    return panels
