import platform
from importlib import metadata
from pathlib import Path

from tabulate import tabulate

from config import RANDOM_STATE
from manuscript_utils import ManuscriptContext, age_table, build_group_shap_share_df, genomic_masking, inventory


def genomic_masking_text(dm):
    n = genomic_masking(dm)
    return (f"Genomic features are masked for {n['masked_lines']:,} of {n['lines']:,} modeled mLoTs, spanning "
            f"{n['masked_patients']:,} of {n['patients']:,} patients (GENOMICS_MISSING = 1: line precedes sequencing).")


def software():
    names = [line.split("==")[0] for line in Path("requirements.txt").read_text().split()]
    return [f"Current local Python: {platform.python_version()}",
            "Historical training Python version is unconfirmed. Installed package versions:",
            "\n".join(f"{name}=={metadata.version(name)}" for name in names)]


def shap_share_table(ctx):
    shares = build_group_shap_share_df(ctx.shap_oof, ctx.group_map)
    means = 100 * shares.mean(axis=0)
    combined = shares[["Imaging coverage and cancer status", "Measurement cadence"]].sum(axis=1)
    means.loc["Imaging coverage + measurement cadence"] = 100 * combined.mean()
    return means.rename("Mean normalized SHAP share, percent").rename_axis("Feature group").reset_index()


def generate(context=None):
    ctx = ManuscriptContext()
    features = inventory()
    dm = ctx.dm
    parts = ["Model inputs", f"Total: {len(features)}. Surveillance-pattern inputs: {features.surveillance_pattern.sum()} (overlapping).",
             tabulate(features.groupby("modality", sort=False).size().rename("Inputs").reset_index(),
                      headers="keys", showindex=False, tablefmt="psql")]
    for kind in ("MUTATION", "AMPLIFICATION", "DELETION"):
        prefix = "GENOMICS_" + kind + "_"
        genes = [f.removeprefix(prefix) for f in features.feature if f.startswith(prefix)]
        parts.append(f"{kind.title()} ({len(genes)}): " + ", ".join(genes) + ".")
    parts.extend(["Main text, genomic masking", genomic_masking_text(dm),
                  "Patient-level table: one record at the first retained line.",
                  tabulate(age_table(dm, True), headers="keys", showindex=False, tablefmt="psql", floatfmt=".1f"),
                  "Line-level rounding check", tabulate(age_table(dm, False), headers="keys", showindex=False,
                  tablefmt="psql", floatfmt=".4f"), "Complete model-input list",
                  tabulate(features, headers="keys", showindex=False, tablefmt="psql"),
                  f"Cross-validation seed: {RANDOM_STATE}. Fold count: 5. Numeric zero filling follows training median/IQR scaling.",
                  *software(),
                  "Extended Data Figure 2: calculated group SHAP shares at 365 days.",
                  "Use the manuscript grouped helper: absolute group sums divided by the sum of absolute group sums for each line, then average across held-out lines.",
                  tabulate(shap_share_table(ctx), headers="keys", showindex=False, tablefmt="psql", floatfmt=".4f"),
                  "Inputs: data/design_matrix.csv; data/features_dict.json; data/feature_groups.json; saved GBSA shap_values.npz; config.py; requirements.txt; installed package metadata."])
    return "\n\n".join(parts) + "\n"
