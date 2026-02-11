import argparse
import os
from typing import Dict

from data_paths import results_root


def build_common_arg_parser(add_help: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument("--age-range", type=str, default=None)  # format: "lo,hi"
    parser.add_argument("--exclude-modalities", type=str, default=None)  # comma list
    parser.add_argument("--exclude-columns", type=str, default=None)  # comma list
    parser.add_argument("--group-label", type=str, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument(
        "--data-suffix",
        type=str,
        default="",
        help="Suffix appended to data/design-matrix artifacts (e.g., 'split2y').",
    )
    parser.add_argument("--eval-horizon-days", type=float, default=None)
    parser.add_argument("--explain", dest="explain", action="store_true")
    parser.add_argument("--no-explain", dest="explain", action="store_false")
    parser.add_argument(
        "--save-survival-curves", dest="save_survival_curves", action="store_true"
    )
    parser.add_argument(
        "--save-full-survival",
        dest="save_full_survival",
        action="store_true",
        help="Persist full survival curves (n_test × time_grid).",
    )
    parser.add_argument(
        "--survival-curve-patient-id",
        type=str,
        default=None,
        help="PATIENT_ID used for targeted survival curve visualizations.",
    )
    parser.set_defaults(explain=None, save_survival_curves=False)
    return parser


def _parse_range(spec: str | None):
    if not spec:
        return None
    try:
        lo, hi = spec.split(",")
        return (float(lo), float(hi))
    except Exception:
        return None


def _parse_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def _parse_bool_env(value: str | None) -> bool | None:
    if value is None:
        return None
    txt = value.strip().lower()
    if txt in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if txt in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def resolve_common_args(args: argparse.Namespace) -> Dict[str, object]:
    age_range = _parse_range(args.age_range or os.getenv("AGE_RANGE"))
    exclude_modalities = _parse_csv(
        args.exclude_modalities or os.getenv("EXCLUDE_MODALITIES")
    )
    exclude_columns = _parse_csv(args.exclude_columns or os.getenv("EXCLUDE_COLUMNS"))
    group_label = args.group_label or os.getenv("GROUP_LABEL")

    random_state = args.random_state
    if random_state is None:
        env_seed = os.getenv("RANDOM_STATE")
        if env_seed is not None:
            try:
                random_state = int(env_seed)
            except ValueError:
                random_state = None

    env_explain = _parse_bool_env(os.getenv("EXPLAIN"))
    if args.explain is None:
        explain = True if env_explain is None else bool(env_explain)
    else:
        explain = bool(args.explain)

    eval_horizon_days = args.eval_horizon_days
    if eval_horizon_days is None and os.getenv("EVAL_HORIZON_DAYS"):
        try:
            eval_horizon_days = float(os.getenv("EVAL_HORIZON_DAYS"))  # type: ignore[arg-type]
        except ValueError:
            eval_horizon_days = None

    env_save_curves = _parse_bool_env(os.getenv("SAVE_SURVIVAL_CURVES"))
    if args.save_survival_curves:
        save_survival_curves = True
    elif env_save_curves is not None:
        save_survival_curves = bool(env_save_curves)
    else:
        save_survival_curves = False
    save_full_survival = bool(getattr(args, "save_full_survival", False))

    survival_curve_patient_id = getattr(args, "survival_curve_patient_id", None)
    if survival_curve_patient_id is None:
        survival_curve_patient_id = os.getenv("SURVIVAL_CURVE_PATIENT_ID")
    if survival_curve_patient_id is not None:
        survival_curve_patient_id = str(survival_curve_patient_id).strip() or None

    data_suffix = (args.data_suffix or os.getenv("DATA_SUFFIX") or "").strip()
    root_base = args.results_root or os.getenv("RESULTS_ROOT") or "results"
    resolved_results_root = results_root(root_base, data_suffix)

    return {
        "age_range": age_range,
        "exclude_modalities": exclude_modalities,
        "exclude_columns": exclude_columns,
        "group_label": group_label,
        "random_state": random_state,
        "results_root": resolved_results_root,
        "explain": explain,
        "eval_horizon_days": eval_horizon_days,
        "save_survival_curves": bool(save_survival_curves),
        "save_full_survival": save_full_survival,
        "data_suffix": data_suffix,
        "survival_curve_patient_id": survival_curve_patient_id,
    }
