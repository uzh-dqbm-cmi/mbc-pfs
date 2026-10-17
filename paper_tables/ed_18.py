import pandas as pd

from manuscript_utils import inventory
from paper_tables.common import Panel

NUMBER = "Extended Data Table 18"
LABEL = "table:model-inputs"
TITLE = "Complete model input list"
GENOMIC_KINDS = {"mutation": "GENOMICS_MUTATION_", "amplification": "GENOMICS_AMPLIFICATION_",
                 "deletion": "GENOMICS_DELETION_", "PD-L1/MMR": ("PDL1_", "MMR_"), "missingness": "GENOMICS_MISSING"}


def build(ctx):
    features = inventory()
    counts = features.groupby("modality", sort=False).size()
    panels = []
    for modality, part in features.groupby("modality", sort=False):
        table = pd.DataFrame({"Input name": part.feature, "Feature group": part.feature_group,
                              "SP": part.surveillance_pattern.map({True: "Yes", False: "No"})})
        panels.append(Panel(f"{modality} ({len(part)} inputs)", table))
    genomic = features.feature[features.modality == "Genomics"]
    panels[-1].note = "\n".join([
        f"{len(features)} inputs: " + ", ".join(f"{n} {m.lower()}" for m, n in counts.items()) + ".",
        f"SP: surveillance-pattern ablation input ({int(features.surveillance_pattern.sum())} inputs, overlapping the modalities).",
        "Genomic inputs: " + ", ".join(f"{int(genomic.str.startswith(p).sum())} {k}" for k, p in GENOMIC_KINDS.items()) + "."])
    return panels
