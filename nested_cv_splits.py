from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (StratifiedGroupKFold,
                                     StratifiedShuffleSplit)

from config import (
    C_INDEX,
    EVENT_COL,
    FEATURES_DICT_PATH,
    LINE_COL,
    LINE_START_WINDOW_DAYS,
    MODALITY_GROUPS,
    PATIENT_ID_COL,
    RESULTS_PATH,
    TIME_COL,
)
from cv_config import (
    ABLATION_CONFIGS_PATH,
    C_SELECTION_EPS,
    K_INNER,
    K_OUTER,
    MODEL_CONFIGS,
    RANDOM_STATE,
    SPLITS_OUTPUT_DIR,
)
from data_paths import data_dir, window_results_root
from preprocess import _hr_her2_admin_strata

GROUP_COL = PATIENT_ID_COL


def _min_groups_per_stratum(strata: np.ndarray, groups: np.ndarray) -> int:
    counts: Dict[str, set] = {}
    for s, g in zip(strata, groups):
        counts.setdefault(str(s), set()).add(str(g))
    if not counts:
        return 0
    return min(len(v) for v in counts.values())


def generate_early_stopping_split(
    design_matrix: pd.DataFrame,
    seed: int,
    val_frac: float,
    base_indices: np.ndarray,
) -> Dict[str, np.ndarray]:
    work_df = design_matrix.iloc[base_indices]
    groups_all = work_df[GROUP_COL].astype(str).to_numpy()
    strata = _hr_her2_admin_strata(
        work_df,
        group_col=GROUP_COL,
        time_col=TIME_COL,
        event_col=EVENT_COL,
    )
    if len(groups_all) != len(strata):
        raise ValueError("Strata and design matrix length mismatch.")

    min_per_stratum = _min_groups_per_stratum(strata, groups_all)
    if min_per_stratum < 2:
        raise ValueError(
            "Insufficient groups per stratum for stratified split: "
            f"need at least 2, found {min_per_stratum}."
        )

    group_labels = []
    strata_labels = []
    for g, s in zip(groups_all, strata):
        if g not in group_labels:
            group_labels.append(g)
            strata_labels.append(s)
    group_labels_np = np.asarray(group_labels, dtype=str)
    strata_labels_np = np.asarray(strata_labels, dtype=str)
    if group_labels_np.size < 2:
        raise ValueError(
            "Insufficient groups for early-stopping split: need at least 2."
        )

    holdout_n = int(round(float(val_frac) * float(group_labels_np.size)))
    holdout_n = max(1, min(int(group_labels_np.size - 1), holdout_n))
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=holdout_n, random_state=seed
    )
    _, group_holdout_local = next(splitter.split(group_labels_np, strata_labels_np))
    holdout_groups = set(group_labels_np[group_holdout_local])

    mask_holdout = work_df[GROUP_COL].astype(str).isin(holdout_groups).to_numpy()
    holdout_idx = base_indices[mask_holdout]
    train_idx = base_indices[~mask_holdout]

    return {
        "train_idx": np.asarray(train_idx, dtype=int),
        "early_stop_idx": np.asarray(holdout_idx, dtype=int),
    }


def generate_nested_cv_indices(
    design_matrix: pd.DataFrame,
    seed: int,
    output_dir: Path,
) -> Tuple[Dict[str, object], List[Path]]:
    if design_matrix.empty:
        raise ValueError("Design matrix is empty; cannot build nested CV splits.")

    required_cols = {GROUP_COL, TIME_COL, EVENT_COL, "LINE"}
    missing = [col for col in required_cols if col not in design_matrix.columns]
    if missing:
        raise KeyError(f"Design matrix missing required columns: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "seed": int(seed),
        "k_outer": int(K_OUTER),
        "k_inner": int(K_INNER),
        "splits": [],
    }
    split_paths: List[Path] = []

    groups_all = design_matrix[GROUP_COL].astype(str).to_numpy()
    strata = _hr_her2_admin_strata(
        design_matrix,
        group_col=GROUP_COL,
        time_col=TIME_COL,
        event_col=EVENT_COL,
    )

    if len(groups_all) != len(strata):
        raise ValueError("Strata and design matrix length mismatch.")
    unique = np.unique(strata)
    assert len(unique) == 8, f"Expected 8 strata, found {len(unique)}."

    min_per_stratum = _min_groups_per_stratum(strata, groups_all)
    needed = max(K_OUTER, K_INNER)
    if min_per_stratum < needed:
        raise ValueError(
            "Insufficient groups per stratum for requested folds: "
            f"need at least {needed}, found {min_per_stratum}."
        )

    outer_cv = StratifiedGroupKFold(n_splits=K_OUTER, shuffle=True, random_state=seed)

    for outer_idx, (train_val_idx, test_idx) in enumerate(
        outer_cv.split(design_matrix, strata, groups=groups_all)
    ):
        train_val_idx = np.asarray(train_val_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        outer_es_split = generate_early_stopping_split(
            design_matrix=design_matrix,
            seed=seed,
            val_frac=0.10,
            base_indices=train_val_idx,
        )
        outer_es_idx = outer_es_split["early_stop_idx"]
        y_outer = strata[train_val_idx]
        groups_outer = groups_all[train_val_idx]
        X_outer = design_matrix.iloc[train_val_idx]

        inner_cv = StratifiedGroupKFold(
            n_splits=K_INNER,
            shuffle=True,
            random_state=seed,
        )

        for inner_idx, (inner_train_local, inner_val_local) in enumerate(
            inner_cv.split(X_outer, y_outer, groups=groups_outer)
        ):
            idx_train = train_val_idx[np.asarray(inner_train_local, dtype=int)]
            idx_val = train_val_idx[np.asarray(inner_val_local, dtype=int)]
            idx_test = test_idx.copy()
            inner_es_split = generate_early_stopping_split(
                design_matrix=design_matrix,
                seed=seed,
                val_frac=0.10,
                base_indices=idx_train,
            )
            inner_es_idx = inner_es_split["early_stop_idx"]

            def _rows(name: str, idxs: np.ndarray) -> pd.DataFrame:
                idxs = np.asarray(idxs, dtype=int)
                subset = design_matrix.iloc[idxs]
                inner_es_mask = np.isin(idxs, inner_es_idx)
                outer_es_mask = np.isin(idxs, outer_es_idx)
                return pd.DataFrame(
                    {
                        "split": name,
                        "idx": idxs,
                        "PATIENT_ID": subset[GROUP_COL].astype(str).to_numpy(),
                        "LINE": subset["LINE"].astype(int).to_numpy(),
                        "outer_fold": int(outer_idx),
                        "inner_fold": int(inner_idx),
                        "inner_es": inner_es_mask.astype(int),
                        "outer_es": outer_es_mask.astype(int),
                    }
                )

            split_df = pd.concat(
                [
                    _rows("train", idx_train),
                    _rows("val", idx_val),
                    _rows("test", idx_test),
                ],
                ignore_index=True,
            )

            split_filename = (
                f"nestedcv_seed{seed}_outer{outer_idx}_inner{inner_idx}.csv"
            )
            split_path = output_dir / split_filename
            if split_path.exists():
                if not split_df.equals(pd.read_csv(split_path)):
                    raise ValueError(
                        f"Existing split file {split_path} differs from newly generated split."
                    )
            else:
                split_df.to_csv(split_path, index=False)
            split_paths.append(split_path)

            manifest[f"outer{outer_idx}_inner{inner_idx}"] = {
                "outer_fold": int(outer_idx),
                "inner_fold": int(inner_idx),
                "path": split_path.as_posix(),
                "n_train": int(idx_train.size),
                "n_val": int(idx_val.size),
                "n_test": int(idx_test.size),
            }

        test_groups = set(
            design_matrix.iloc[idx_test][GROUP_COL].astype(str).to_numpy()
        )
        val_groups = set(design_matrix.iloc[idx_val][GROUP_COL].astype(str).to_numpy())
        train_groups = set(
            design_matrix.iloc[idx_train][GROUP_COL].astype(str).to_numpy()
        )
        assert (
            test_groups.isdisjoint(val_groups)
            and test_groups.isdisjoint(train_groups)
            and val_groups.isdisjoint(train_groups)
        ), "Groups are not disjoint between train, val, and test sets."

    manifest_path = output_dir / f"nestedcv_manifest_seed{seed}.json"
    with manifest_path.open("w") as fout:
        json.dump(manifest, fout, indent=2)

    return manifest, split_paths


def add_run_args(parser: argparse.ArgumentParser, default_models: str) -> None:
    parser.add_argument(
        "--models",
        type=str,
        default=default_models,
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
    parser.add_argument(
        "--window-days",
        type=int,
        default=LINE_START_WINDOW_DAYS,
        help=(
            "Treatment window of the design matrix: other values than the default read "
            "data/window_<N>d/ and write splits and results to results_window/<N>d/."
        ),
    )


def _modality_columns(dm: pd.DataFrame, modality: str) -> List[str]:
    if modality == "Surveillance Patterns":
        return [c for c in dm.columns if c in MODALITY_GROUPS["Surveillance Patterns"]]
    features = json.load(open(FEATURES_DICT_PATH))
    return sorted(
        {
            col
            for group in MODALITY_GROUPS[modality]
            for col in features[group]
            if col in dm.columns
        }
    )


def _write_columns(path: Path, cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fout:
        fout.write(", ".join(cols) + "\n")


def prepare_run(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, object], Path, Optional[Dict[str, List[str]]]]:
    n_ablations = sum(
        (args.exclude_modality is not None, args.only_modality is not None, args.drop_age)
    )
    if n_ablations > 1:
        raise SystemExit("Only one of --exclude-modality, --only-modality, --drop-age can be set.")
    window_run = args.window_days != LINE_START_WINDOW_DAYS
    if window_run and (n_ablations or args.no_treatment_masking):
        raise SystemExit("--window-days runs the masked full-model nested CV only.")

    dm = pd.read_csv(data_dir(args.window_days) / "design_matrix.csv")
    dm = dm[dm[EVENT_COL] != -1].reset_index(drop=True)
    if window_run:
        results_root = window_results_root(args.window_days)
        splits_dir = results_root / "cv_splits"
    else:
        splits_dir = Path(SPLITS_OUTPUT_DIR)
        if args.no_treatment_masking:
            results_root = Path("leakage")
        else:
            results_root = Path("results_ablation") if n_ablations else RESULTS_PATH
    manifest, _ = generate_nested_cv_indices(
        design_matrix=dm, seed=RANDOM_STATE, output_dir=splits_dir
    )
    if not n_ablations:
        return dm, manifest, results_root, None

    ablation_configs = json.load(open(ABLATION_CONFIGS_PATH))
    print(f"Loaded ablation config names from {ABLATION_CONFIGS_PATH}.")
    if args.exclude_modality:
        exclude_cols = _modality_columns(dm, args.exclude_modality)
        dm = dm.drop(columns=exclude_cols)
        results_root = results_root / f"exclude_{args.exclude_modality.lower()}"
        _write_columns(results_root / "excluded_columns.txt", exclude_cols)
    elif args.only_modality:
        include_cols = _modality_columns(dm, args.only_modality) + [
            EVENT_COL,
            TIME_COL,
            PATIENT_ID_COL,
            LINE_COL,
        ]
        dm = dm[include_cols]
        print(f"Kept only cols {include_cols} for modality {args.only_modality}")
        results_root = results_root / f"only_{args.only_modality.lower()}"
        _write_columns(results_root / "included_columns.txt", include_cols)
    else:
        dm = dm.drop(columns=["AGE"])
        print("Dropped AGE column from design matrix.")
        results_root = results_root / "drop_age"
    return dm, manifest, results_root, ablation_configs


def model_configs(
    model: str, ablation_configs: Optional[Dict[str, List[str]]]
) -> Dict[str, dict]:
    configs = MODEL_CONFIGS[model]
    if ablation_configs is not None:
        allowed = set(ablation_configs.get(model, []))
        configs = [cfg for cfg in configs if cfg.get("name") in allowed]
        if len(configs) != 1:
            raise ValueError(
                f"Expected exactly one ablation config for model {model} in {ABLATION_CONFIGS_PATH}, got {len(configs)}."
            )
        print(f"Using single ablation config {configs[0]['name']} for {model}; skipping inner folds.")
    return {cfg["name"]: cfg for cfg in configs}


def select_best_config(
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


def write_metrics(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fout:
        json.dump(payload, fout, indent=2)
