import textwrap

from manuscript_utils import feature_group_rules
from paper_tables.common import Panel

NUMBER = "Extended Data Table 16"
LABEL = "table:feature-groups"
TITLE = "Feature-group selection rules used for grouped attribution analyses"


def build(ctx):
    rules = feature_group_rules()
    rules["Selection rule"] = [textwrap.fill(rule, 80, break_long_words=False, break_on_hyphens=False)
                               for rule in rules["Selection rule"]]
    return [Panel("", rules)]
