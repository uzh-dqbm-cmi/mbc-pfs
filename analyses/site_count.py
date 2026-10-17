from tabulate import tabulate

from analyses.common.ibs_bootstrap import paired
from analyses.common.survival import SurvivalContext, site_columns

ORIGINAL, SITE_COUNT = "full_gbsa", "gbsa_site_count"
MODELS = {ORIGINAL: "Original GBSA", SITE_COUNT: "GBSA plus site count"}


def site_count(ctx):
    return ctx.pooled(MODELS), paired(ctx, SITE_COUNT, ORIGINAL)


def generate(context=None):
    ctx = context or SurvivalContext()
    pooled, diff = site_count(ctx)
    rows = [[MODELS[r.Index], f"{r.C:.5f}", f"{r.C_low:.5f}–{r.C_high:.5f}", f"{r.IBS:.5f}"] for r in pooled.itertuples()]
    return "\n\n".join([
        "Number of metastatic-site categories",
        f"Cohort: {len(ctx.dm):,} lines; {len(ctx.ids):,} patients.",
        f"Added input: sum of {len(site_columns(ctx.dm))} recorded site-category flags in the 90 days before line start.",
        "C and IBS use pooled out-of-fold predictions. C intervals use 1,000 paired patient draws, seed 42.",
        tabulate(rows, headers=["Model", "Pooled C", "95% patient-bootstrap CI", "Pooled IBS"], tablefmt="psql", disable_numparse=True),
        tabulate([["C, site count minus original", f"{diff.delta_C:+.5f}", f"{diff.CI_low:+.5f} to {diff.CI_high:+.5f}"],
                  ["IBS, site count minus original", f"{diff.delta_IBS:+.5f}", f"{diff.IBS_low:+.6f} to {diff.IBS_high:+.6f}"]],
                 headers=["Pooled comparison", "Change", "95% paired patient-bootstrap CI"], tablefmt="psql", disable_numparse=True),
        "New IBS interval: censoring KM and the observed censoring-time support are recalculated within each patient draw. The five fitted models and their selected settings stay fixed."
    ]) + "\n"


if __name__ == "__main__":
    print(generate())
