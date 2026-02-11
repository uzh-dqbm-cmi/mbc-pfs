from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    AUC,
    C_INDEX,
    DESIGN_MATRIX_PATH,
    EVENT_COL,
    FEATURES_DICT_PATH,
    IBS,
    LINE_COL,
    MODALITY_GROUPS,
    PATIENT_ID_COL,
    RESULTS_PATH,
    TIME_COL,
)
from cv_config import MODEL_CONFIGS, RANDOM_STATE, SPLITS_OUTPUT_DIR
from deephit import train_deephit
from deepsurv import train_deepsurv
from dryrun import train_dryrun
from model_setup import EVENT_COL
from nested_cv_splits import generate_nested_cv_indices
from utils import save_eval_metrics

MODEL_RUNNERS = {
    "deepsurv": train_deepsurv,
    "deephit": train_deephit,
    "dryrun": train_dryrun,
}
DRY_RUN_METRIC = 0.5
# Treat configs within this C-index margin of the best score as ties.
C_SELECTION_EPS = 0.003


def _write_metrics_artifact(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fout:
        json.dump(payload, fout, indent=2)


def _select_best_config(
    config_scores: Dict[str, Dict[str, float]],
    rank_of: Dict[str, int],
) -> tuple[str, float, list[str]]:
    best_c = max(float(metrics[C_INDEX]) for metrics in config_scores.values())
    tied_names = sorted(
        (
            name
            for name, metrics in config_scores.items()
            if float(metrics[C_INDEX]) >= (best_c - C_SELECTION_EPS)
        ),
        key=lambda name: rank_of[name],
    )
    return tied_names[0], best_c, tied_names


def main():
    parser = argparse.ArgumentParser(
        description="Nested CV orchestrator that calls in-process trainers."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="deepsurv",
        help="Comma-separated model list.",
    )
    parser.add_argument(
        "--exclude-modality",
        type=str,
        default=None,
        help="Name of modality group to exclude.",
    )
    parser.add_argument(
        "--only-modality",
        type=str,
        default=None,
        help="Name of modality group to use exclusively.",
    )
    parser.add_argument(
        "--drop-age",
        action="store_true",
        help="Drop AGE column from design matrix.",
    )
    parser.add_argument(
        "--no-treatment-masking",
        action="store_true",
        help="Disable treatment masking by dropping PLANNED_ instead of default prefixes.",
    )
    args = parser.parse_args()
    ignore_prefix = "PLANNED_" if args.no_treatment_masking else None

    dm = pd.read_csv(DESIGN_MATRIX_PATH)
    dm = dm[dm[EVENT_COL] != -1].reset_index(drop=True)

    manifest, _ = generate_nested_cv_indices(
        design_matrix=dm, seed=RANDOM_STATE, output_dir=Path(SPLITS_OUTPUT_DIR)
    )

    run_ablation = bool(args.exclude_modality or args.only_modality or args.drop_age)
    if args.no_treatment_masking:
        results_root = Path("leakage")
    else:
        results_root = Path("results_ablation") if run_ablation else RESULTS_PATH
    ablation_cfg_names: Dict[str, List[str]] = {}
    ablation_cfg_path = Path("data/ablation_configs.json")
    if run_ablation:
        if not ablation_cfg_path.exists():
            raise FileNotFoundError(
                f"Missing ablation config file: {ablation_cfg_path}"
            )
        try:
            payload = json.load(open(ablation_cfg_path))
            ablation_cfg_names = {
                str(model): [str(name) for name in names]
                for model, names in payload.items()
                if isinstance(names, list)
            }
            print(f"Loaded ablation config names from {ablation_cfg_path}.")
        except Exception as exc:
            raise ValueError(f"Failed to load {ablation_cfg_path}: {exc}") from exc

    if run_ablation:
        FEATURES = json.load(open(FEATURES_DICT_PATH))
        # only one of exclude_modality, only_modality, drop_age should be set
        assert (
            sum(
                (
                    args.exclude_modality is not None,
                    args.only_modality is not None,
                    args.drop_age,
                )
            )
            <= 1
        ), "Only one of --exclude-modality, --only-modality, --drop-age can be set."

        if args.exclude_modality:
            if args.exclude_modality == "Surveillance Patterns":
                exclude_cols = [
                    c
                    for c in dm.columns
                    if c in MODALITY_GROUPS["Surveillance Patterns"]
                ]
            else:
                exclude_groups = MODALITY_GROUPS[args.exclude_modality]
                exclude_cols = sorted(
                    {
                        col
                        for group in exclude_groups
                        for col in FEATURES[group]
                        if col in dm.columns
                    }
                )
            dm = dm.drop(columns=exclude_cols)
            results_root = results_root / f"exclude_{args.exclude_modality.lower()}"
            # dump a text line saying which columns were dropped to results_root/excluded_columns.txt
            excluded_cols_path = results_root / "excluded_columns.txt"
            excluded_cols_path.parent.mkdir(parents=True, exist_ok=True)
            with open(excluded_cols_path, "w") as fout:
                fout.write(", ".join(exclude_cols) + "\n")

        elif args.only_modality:
            if args.only_modality == "Surveillance Patterns":
                include_cols = [
                    c
                    for c in dm.columns
                    if c in MODALITY_GROUPS["Surveillance Patterns"]
                ]
            else:
                include_groups = MODALITY_GROUPS[args.only_modality]
                include_cols = sorted(
                    {
                        col
                        for group in include_groups
                        for col in FEATURES[group]
                        if col in dm.columns
                    }
                )

            include_cols += [
                EVENT_COL,
                TIME_COL,
                PATIENT_ID_COL,
                LINE_COL,
            ]
            dm = dm[include_cols]
            print(f"Kept only cols {include_cols} for modality {args.only_modality}")
            results_root = results_root / f"only_{args.only_modality.lower()}"
            # dump a text line saying which columns were dropped to results_root/excluded_columns.txt
            included_cols_path = results_root / "included_columns.txt"
            included_cols_path.parent.mkdir(parents=True, exist_ok=True)
            with open(included_cols_path, "w") as fout:
                fout.write(", ".join(include_cols) + "\n")
        elif args.drop_age:
            dm = dm.drop(columns=["AGE"])
            print("Dropped AGE column from design matrix.")
            results_root = results_root / "drop_age"

    for raw_model in args.models.split(","):
        try:
            model = raw_model.strip()
            configs = MODEL_CONFIGS[model]
            if run_ablation:
                allowed = set(ablation_cfg_names.get(model, []))
                if not allowed:
                    raise ValueError(
                        f"No ablation config names listed for model {model} in {ablation_cfg_path}."
                    )
                configs = [cfg for cfg in configs if cfg.get("name") in allowed]
                if not configs:
                    raise ValueError(
                        f"No configs left for model {model} after ablation filter."
                    )
                print(
                    f"Using {len(configs)} filtered configs for {model} from {ablation_cfg_path}."
                )
            configs_dict = {cfg["name"]: cfg for cfg in configs}
            model_metrics: Dict[str, object] = {
                "model": model,
                "outer_folds": {},
            }
            if run_ablation:
                if len(configs_dict) != 1:
                    raise ValueError(
                        f"Expected exactly one ablation config for model {model}, got {len(configs_dict)}."
                    )
                config_name, config = next(iter(configs_dict.items()))
                print(
                    f"Using single ablation config {config_name} for {model}; skipping inner folds."
                )
                for outer_num in range(manifest["k_outer"]):
                    print(f"OUTER FOLD {outer_num}: {model}--{config_name}")
                    outer_key = f"outer{outer_num}"
                    outer_entry: Dict[str, object] = {
                        "outer": outer_num,
                        "inner_skipped": True,
                    }
                    split = pd.read_csv(manifest[f"outer{outer_num}_inner0"]["path"])
                    if "outer_es" not in split.columns:
                        raise ValueError(
                            f"Split file {manifest[f'outer{outer_num}_inner0']['path']} missing `outer_es` column."
                        )
                    non_test_mask = split["split"] != "test"
                    outer_es_mask = split["outer_es"].astype(bool)
                    train_idx = split.loc[
                        non_test_mask & ~outer_es_mask, "idx"
                    ].to_numpy(dtype=int)
                    early_stop_idx = split.loc[
                        non_test_mask & outer_es_mask, "idx"
                    ].to_numpy(dtype=int)
                    test_idx = split[split["split"] == "test"]["idx"].to_numpy(
                        dtype=int
                    )
                    if early_stop_idx.size == 0:
                        raise ValueError(
                            f"Split file {manifest[f'outer{outer_num}_inner0']['path']} produced empty `outer_es` set."
                        )
                    if test_idx.size == 0:
                        raise ValueError(
                            f"Split file {manifest[f'outer{outer_num}_inner0']['path']} produced empty outer `test` set."
                        )
                    final_config = config.copy()
                    final_config["outer_fold"] = outer_num
                    final_config["ignore_prefix"] = ignore_prefix
                    outer_outdir = results_root / model / f"outer{outer_num}"
                    try:
                        final_metrics = MODEL_RUNNERS[model](
                            train_idx=train_idx,
                            val_idx=early_stop_idx,
                            test_idx=test_idx,
                            design_matrix=dm,
                            config=final_config,
                            outer=True,
                            outdir=outer_outdir / config_name,
                            run_ablation=run_ablation,
                        )
                        print(f"  FINAL EVAL {C_INDEX}: {final_metrics.get(C_INDEX)}")
                        print(f"  FINAL EVAL {AUC}: {final_metrics.get(AUC)}")
                        print(f"  FINAL EVAL {IBS}: {final_metrics.get(IBS)}")
                    except Exception as exc:
                        print(f"[ERROR] {model} outer{outer_num} final: {exc}")
                        traceback.print_exc()
                        final_metrics = {"error": str(exc)}
                    best_model_record = {
                        "split_path": manifest[f"outer{outer_num}_inner0"]["path"],
                        "results_path": str(outer_outdir / config_name),
                        "config": config_name,
                        "metrics": final_metrics,
                        "name": config_name,
                    }
                    outer_entry["best_model"] = best_model_record
                    model_metrics["outer_folds"][outer_key] = best_model_record
                    _write_metrics_artifact(outer_outdir / "fold_metrics.json", outer_entry)
                _write_metrics_artifact(
                    results_root / model / "eval_metrics.json", model_metrics
                )
                print(f"Wrote metrics to {results_root / model / 'eval_metrics.json'}")
                continue

            for outer_num in range(manifest["k_outer"]):
                print(f"OUTER FOLD {outer_num}: {model}")
                outer_key = f"outer{outer_num}"
                outer_entry: Dict[str, object] = {"outer": outer_num}
                model_metrics["outer_folds"][
                    outer_key
                ] = None  # placeholder; fill after best_model
                config_avgs: Dict[str, Dict[str, float]] = {}
                inner_data: Dict[int, Dict[str, object]] = {
                    i: {} for i in range(manifest["k_inner"])
                }

                for config_name, config in configs_dict.items():
                    c_vals: List[float] = []
                    auc_vals: List[float] = []
                    ibs_vals: List[float] = []
                    for inner_num in range(manifest["k_inner"]):
                        fold_label = f"outer{outer_num}_inner{inner_num}"
                        split = pd.read_csv(manifest[fold_label]["path"])
                        train_mask = split["split"] == "train"
                        inner_es_mask = split["inner_es"].astype(bool)
                        train_idx = split.loc[
                            train_mask & ~inner_es_mask, "idx"
                        ].to_numpy()
                        early_stop_idx = split.loc[
                            train_mask & inner_es_mask, "idx"
                        ].to_numpy()
                        eval_idx = split.loc[split["split"] == "val", "idx"].to_numpy()
                        print(f"  INNER FOLD {inner_num}: {model}--{config_name}")
                        config_ctx = dict(config)
                        config_ctx.setdefault("seed", RANDOM_STATE)
                        config_ctx["outer_fold"] = outer_num
                        config_ctx["inner_fold"] = inner_num
                        config_ctx["ignore_prefix"] = ignore_prefix
                        inner_outdir = (
                            results_root
                            / model
                            / f"outer{outer_num}"
                            / f"inner{inner_num}"
                            / config_name
                        )
                        inner_outdir_str = str(inner_outdir)
                        if early_stop_idx.size == 0:
                            raise ValueError(
                                f"Split file {manifest[fold_label]['path']} produced empty `inner_es` set."
                            )
                        if eval_idx.size == 0:
                            raise ValueError(
                                f"Split file {manifest[fold_label]['path']} produced empty inner `val` set."
                            )
                        try:
                            metrics = MODEL_RUNNERS[model](
                                train_idx=train_idx.astype(int),
                                val_idx=early_stop_idx.astype(int),
                                test_idx=eval_idx.astype(
                                    int
                                ),  # report inner performance on held-out eval fold
                                design_matrix=dm,
                                config=config_ctx,
                                outer=False,
                                outdir=inner_outdir,
                                run_ablation=run_ablation,
                            )
                        except Exception as exc:
                            print(f"[ERROR] {model} {config_name} {fold_label}: {exc}")
                            traceback.print_exc()
                            metrics = {"error": str(exc)}
                            save_eval_metrics(
                                outdir=inner_outdir,
                                metrics=metrics,
                            )
                        inner_data[inner_num][config_name] = {
                            "results_path": inner_outdir_str,
                            "metrics": metrics,
                        }
                        c_vals.append(float(metrics[C_INDEX]))
                        auc_vals.append(float(metrics[AUC]))
                        ibs_vals.append(float(metrics[IBS]))
                        print(
                            f"    {C_INDEX}: {metrics.get(C_INDEX)}, {AUC}: {metrics.get(AUC)}, {IBS}: {metrics.get(IBS)}\n"
                        )

                    config_avgs[config_name] = {
                        C_INDEX: np.mean(c_vals),
                        AUC: np.mean(auc_vals),
                        IBS: np.mean(ibs_vals),
                    }

                # select the best model, fit on the outer fold. Save models, metrics, surv outputs, and SHAP values.
                rank_of = {
                    name: cfg["internal_ranking"] for name, cfg in configs_dict.items()
                }
                best_config_name, best_c, c_tie_names = _select_best_config(
                    config_avgs,
                    rank_of,
                )
                print(
                    "  Tie-band selection:"
                    f" best_C={best_c:.6f},"
                    f" eps={C_SELECTION_EPS:.3f},"
                    f" candidates={c_tie_names}"
                )
                # use the inner0 split to get train/test indices for final eval from outer fold
                split = pd.read_csv(manifest[f"outer{outer_num}_inner0"]["path"])
                if "outer_es" not in split.columns:
                    raise ValueError(
                        f"Split file {manifest[f'outer{outer_num}_inner0']['path']} missing `outer_es` column."
                    )
                non_test_mask = split["split"] != "test"
                outer_es_mask = split["outer_es"].astype(bool)
                train_idx = split.loc[non_test_mask & ~outer_es_mask, "idx"].to_numpy(
                    dtype=int
                )
                early_stop_idx = split.loc[
                    non_test_mask & outer_es_mask, "idx"
                ].to_numpy(dtype=int)
                test_idx = split[split["split"] == "test"]["idx"].to_numpy(dtype=int)
                if early_stop_idx.size == 0:
                    raise ValueError(
                        f"Split file {manifest[f'outer{outer_num}_inner0']['path']} produced empty `outer_es` set."
                    )
                if test_idx.size == 0:
                    raise ValueError(
                        f"Split file {manifest[f'outer{outer_num}_inner0']['path']} produced empty outer `test` set."
                    )
                final_config = configs_dict[best_config_name].copy()
                final_config["outer_fold"] = outer_num
                final_config["ignore_prefix"] = ignore_prefix

                print(f"  OUTER FOLD {outer_num} BEST CONFIG: {best_config_name}")
                print("  Average scores:")
                print(f"    {C_INDEX}: {config_avgs[best_config_name][C_INDEX]}")
                print(f"    {AUC}: {config_avgs[best_config_name][AUC]}")
                print(f"    {IBS}: {config_avgs[best_config_name][IBS]}")
                outer_outdir = results_root / model / f"outer{outer_num}"
                try:
                    final_metrics = MODEL_RUNNERS[model](
                        train_idx=train_idx,
                        val_idx=early_stop_idx,
                        test_idx=test_idx,
                        design_matrix=dm,
                        config=final_config,
                        outer=True,
                        outdir=outer_outdir / best_config_name,
                        run_ablation=run_ablation,
                    )
                    print(f"  FINAL EVAL {C_INDEX}: {final_metrics.get(C_INDEX)}")
                    print(f"  FINAL EVAL {AUC}: {final_metrics.get(AUC)}")
                    print(f"  FINAL EVAL {IBS}: {final_metrics.get(IBS)}")
                except Exception as exc:
                    print(f"[ERROR] {model} outer{outer_num} final: {exc}")
                    traceback.print_exc()
                    final_metrics = {"error": str(exc)}
                best_model_record = {
                    "split_path": manifest[f"outer{outer_num}_inner0"]["path"],
                    "results_path": str(outer_outdir / best_config_name),
                    "config": best_config_name,
                    "metrics": final_metrics,
                }
                for inner_num in range(manifest["k_inner"]):
                    outer_entry[f"inner_{inner_num}"] = inner_data[inner_num]
                for cfg_name, avg in config_avgs.items():
                    outer_entry[f"{cfg_name}_avg"] = avg
                best_model_record["name"] = best_config_name
                outer_entry["best_model"] = best_model_record
                model_metrics["outer_folds"][outer_key] = best_model_record
                _write_metrics_artifact(outer_outdir / "fold_metrics.json", outer_entry)

            _write_metrics_artifact(
                results_root / model / "eval_metrics.json", model_metrics
            )
            print(f"Wrote metrics to {results_root / model / 'eval_metrics.json'}")
        except Exception as exc:
            print(f"[ERROR] {model} overall: {exc}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
