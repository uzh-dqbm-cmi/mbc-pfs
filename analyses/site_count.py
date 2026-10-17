from analyses.common.ibs_bootstrap import paired

ORIGINAL, SITE_COUNT = "full_gbsa", "gbsa_site_count"
MODELS = {ORIGINAL: "Original GBSA", SITE_COUNT: "GBSA plus site count"}


def site_count(ctx):
    return ctx.pooled(MODELS), paired(ctx, SITE_COUNT, ORIGINAL)
