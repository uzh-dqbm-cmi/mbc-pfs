from data_paths import window_results_root
from manuscript_utils import BASE_MODELS, build_model_perf
from paper_tables.common import Panel, fold_performance

NUMBER = "Extended Data Table 11"
LABEL = "table:sensitivity-14d-performance"
TITLE = "Global performance metrics under the 14-day treatment-window sensitivity analysis"

DAYS = 14


def window_performance(days):
    root = window_results_root(days)
    missing = [model for model in BASE_MODELS if not (root / model / "eval_metrics.json").exists()]
    if missing:
        raise FileNotFoundError(f"{root} lacks {', '.join(missing)}: run python pipeline_overview.py --window-days "
                                f"{days}; python cpu_nested_cv.py --window-days {days} --models coxph,gbsa,rsf; "
                                f"python dl_nested_cv.py --window-days {days} --models deepsurv,deephit.")
    return fold_performance(build_model_perf(root, BASE_MODELS)[0])


def build(ctx):
    return [Panel("", window_performance(DAYS))]
