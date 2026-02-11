from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import restricted_mean_survival_time
from matplotlib import cycler
from matplotlib.lines import Line2D

from config import (
    LINE_COL,
    PATIENT_ID_COL,
    PFS_TIME_BIN_EDGES,
    RANDOM_STATE,
)
from feature_groups import feature_module
from survival_metrics import km_curve, mae_pred_vs_km, rmst_from_curve

# plotting utilities used by manuscript.py


# Shared color cycle (blue/orange/green theme) used across plots.
PROJECT_COLOR_CYCLE: list[str] = [
    "#1f4e79",  # blue (dark)
    "#d97706",  # orange (dark)
    "#166534",  # green (dark)
    "#5499c7",  # blue (light)
    "#f59e0b",  # orange (light)
    "#22c55e",  # green (light)
]

SURVIVAL_FIGSIZE = (7, 4.5)
SURVIVAL_X_LIMIT = (0, 730)
SURVIVAL_Y_LIMIT = (0, 1.02)
SURVIVAL_TITLE_SIZE = 13
SURVIVAL_LABEL_SIZE = 12
SURVIVAL_TICK_SIZE = 11
SURVIVAL_LINEWIDTH = 2.3

STYLE = {
    "figure.dpi": 200,
    "savefig.dpi": 400,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica:italic",
    "mathtext.bf": "Helvetica:bold",
    "mathtext.sf": "Helvetica",
    "mathtext.tt": "Helvetica",
    "mathtext.default": "regular",
    "mathtext.fallback": "stixsans",
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 12,
    "axes.labelweight": "regular",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#2f2f2f",
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.prop_cycle": cycler("color", PROJECT_COLOR_CYCLE),
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.color": "#2f2f2f",
    "ytick.color": "#2f2f2f",
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "lines.linewidth": 2.0,
}

# Risk stratification constants (moved from risk_figures.py)
EVENT_COL = "PFS_EVENT"
TIME_COL = "PFS_TIME_DAYS"
HR_COL = "HR"
HER2_COL = "HER2"

STRATA_LABELS: tuple[str, str, str] = ("low", "mid", "high")
STRATA_COLORS: dict[str, str] = {
    "low": PROJECT_COLOR_CYCLE[0],  # blue
    "mid": PROJECT_COLOR_CYCLE[1],  # orange
    "high": PROJECT_COLOR_CYCLE[2],  # green
}
SUBTYPE_ORDER: tuple[str, ...] = (
    "HR-/HER2-",
    "HR-/HER2+",
    "HR+/HER2-",
    "HR+/HER2+",
)


def plot_survival_curves_by_index(
    time_grid: Sequence[float],
    surv_np: np.ndarray,
    indices: Sequence[int],
    output_path: Union[str, Path],
    design_matrix: pd.DataFrame | None = None,
    highlight: Sequence[int] | None = None,
    title: str = "Individual survival curves",
) -> Path:
    """Plot individual survival curves for arbitrary SURV_OOF row indices."""
    plt.rcParams.update(STYLE)
    time_grid = np.asarray(time_grid, dtype=float)
    surv_np = np.asarray(surv_np, dtype=float)
    idx_arr = np.asarray(indices, dtype=int).reshape(-1)
    if surv_np.shape[1] != time_grid.shape[0]:
        raise ValueError("surv_np columns must match time_grid length.")
    if idx_arr.size == 0:
        raise ValueError("indices must contain at least one element.")
    if np.any(idx_arr < 0) or np.any(idx_arr >= surv_np.shape[0]):
        raise IndexError("indices contain out-of-range entries for surv_np.")

    uniq_idx = np.unique(idx_arr)
    curves = surv_np[uniq_idx]
    highlight_set = set(int(x) for x in (highlight or []))

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    color_cycle = PROJECT_COLOR_CYCLE
    for i, global_idx in enumerate(uniq_idx):
        color = color_cycle[i % len(color_cycle)]
        lw = 2.6 if global_idx in highlight_set else 1.9
        alpha = 0.95 if global_idx in highlight_set else 0.82
        label = f"idx {global_idx}"
        if design_matrix is not None:
            row = design_matrix.iloc[int(global_idx)]
            patient = row.get(PATIENT_ID_COL, None)
            lot = row.get(LINE_COL, None)
            if pd.notna(patient):
                label = f"Pt {patient}"
                if pd.notna(lot):
                    label += f" (Line {lot})"
        ax.step(
            time_grid,
            curves[i],
            where="post",
            color=color,
            lw=lw,
            alpha=alpha,
            label=label,
        )

    mean_curve = np.nanmean(curves, axis=0)
    ax.plot(
        time_grid,
        mean_curve,
        color="#111111",
        lw=3.0,
        ls="--",
        label="Mean curve",
        alpha=0.9,
    )

    ax.set_xlabel("Days since index", fontsize=SURVIVAL_LABEL_SIZE)
    ax.set_ylabel("Survival probability", fontsize=SURVIVAL_LABEL_SIZE)
    ax.set_xlim(SURVIVAL_X_LIMIT)
    ax.set_ylim(SURVIVAL_Y_LIMIT)
    ax.set_title(title, fontsize=SURVIVAL_TITLE_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.tick_params(axis="both", labelsize=SURVIVAL_TICK_SIZE)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            frameon=False,
            loc="upper right",
            fontsize=SURVIVAL_TICK_SIZE,
            ncol=2,
        )
    fig.tight_layout()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=800)
    plt.close(fig)
    return out_path


def group_images_2x2(
    image_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    titles: Sequence[str] | None = None,
    figsize: tuple[float, float] = (10.0, 10.0),
    background: str = "#f8f8f8",
) -> Path:
    """Arrange up to four image files into a 2x2 grid."""
    paths = [Path(p) for p in image_paths[:4]]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plt.rcParams.update(STYLE)
    fig.patch.set_facecolor(background)
    flat_axes = axes.ravel()
    titles = list(titles) if titles is not None else [None] * len(paths)

    for i in range(4):
        ax = flat_axes[i]
        ax.axis("off")
        if i >= len(paths):
            ax.set_facecolor(background)
            continue
        path = paths[i]
        if not path.exists():
            ax.text(
                0.5,
                0.5,
                f"Missing:\n{path}",
                ha="center",
                va="center",
                fontsize=9,
            )
            continue
        img = plt.imread(path)
        ax.imshow(img)
        if i < len(titles) and titles[i]:
            ax.set_title(titles[i], fontsize=11, pad=6)

    fig.tight_layout()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400, facecolor=background)
    plt.close(fig)
    return out_path


def _load_shap_payload(npz_path: Path):
    payload = np.load(npz_path, allow_pickle=True)
    shap_values = np.asarray(payload["shap_values"], dtype=float)
    feature_names = [str(x) for x in np.asarray(payload["feature_names"]).tolist()]
    X = np.asarray(payload["X"], dtype=float)
    sample_idx = (
        np.asarray(payload["sample_idx"], dtype=int).reshape(-1)
        if "sample_idx" in payload.files
        else None
    )
    if X.shape[0] != shap_values.shape[0]:
        if (
            sample_idx is not None
            and sample_idx.size >= shap_values.shape[0]
            and np.max(sample_idx[: shap_values.shape[0]]) < X.shape[0]
        ):
            selector = sample_idx[: shap_values.shape[0]]
            X = X[selector]
            sample_idx = selector
        else:
            X = X[: shap_values.shape[0]]
            if sample_idx is not None and sample_idx.size >= shap_values.shape[0]:
                sample_idx = sample_idx[: shap_values.shape[0]]
            else:
                sample_idx = None
    elif sample_idx is not None and sample_idx.size > shap_values.shape[0]:
        sample_idx = sample_idx[: shap_values.shape[0]]
    expected_value = (
        payload["expected_value"] if "expected_value" in payload.files else None
    )
    if expected_value is None:
        raise KeyError(
            f"'expected_value' missing from {npz_path}; rerun SHAP with base values saved."
        )
    base_value = float(np.asarray(expected_value, dtype=float).reshape(-1)[0])
    return shap_values, X, feature_names, base_value, sample_idx


@lru_cache(maxsize=1)
def _load_features_dict(
    path: Path = Path("data/features_dict.json"),
) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; cannot aggregate OHE SHAP features.")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"features_dict.json must be a dict of feature lists, got {type(payload)}."
        )
    normalized: dict[str, list[str]] = {}
    for key, values in payload.items():
        if isinstance(values, list):
            normalized[str(key)] = [str(v) for v in values]
    return normalized


def _build_imaging_ohe_groups(feature_names: list[str]) -> dict[str, list[str]]:
    """
    Group imaging status OHE features such that *_IMAGED_*_STATUS + *_MISSING collapse into one feature.
    """
    imaging_groups: dict[str, list[str]] = defaultdict(list)
    feature_set = set(feature_names)
    for name in feature_names:
        if (
            name.endswith("_IMAGED_Y_STATUS")
            or name.endswith("_IMAGED_N_STATUS")
            or name.endswith("_IMAGED_INDET_STATUS")
        ):
            prefix = name.rsplit("_IMAGED_", 1)[0]
            imaging_groups[prefix].append(name)
    for prefix in list(imaging_groups.keys()):
        missing = f"{prefix}_MISSING"
        if missing in feature_set:
            imaging_groups[prefix].append(missing)
        if len(imaging_groups[prefix]) < 2:
            imaging_groups.pop(prefix, None)
    renamed_groups = {
        f"{prefix}_IMAGING_STATUS": members
        for prefix, members in imaging_groups.items()
    }
    return renamed_groups


def _aggregate_ohe_vectors(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    features_dict_path: Path = Path("data/features_dict.json"),
    extra_groups: Mapping[str, Sequence[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Sum SHAP contributions (and corresponding OHE feature values) for designated one-hot blocks.
    """
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if shap_values.shape[1] != len(feature_names) or X.shape[1] != len(feature_names):
        raise ValueError("Shape mismatch between SHAP arrays and feature_names.")

    features_dict = _load_features_dict(features_dict_path)
    feature_set = set(feature_names)
    group_definitions: dict[str, list[str]] = {}

    def _register_group(group_name: str, members: list[str]):
        filtered = [m for m in members if m in feature_set]
        if len(filtered) >= 2:
            group_definitions[group_name] = filtered

    for block in ("PDL1", "MMR", "DIAGNOSIS"):
        _register_group(block, features_dict.get(block, []))

    imaging_groups = _build_imaging_ohe_groups(feature_names)
    for group_name, members in imaging_groups.items():
        _register_group(group_name, members)
    if extra_groups:
        for name, members in extra_groups.items():
            _register_group(name, list(members))

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    member_to_group: dict[str, str] = {}
    for group_name, members in group_definitions.items():
        for member in members:
            member_to_group[member] = group_name

    aggregated_shap: list[np.ndarray] = []
    aggregated_X: list[np.ndarray] = []
    aggregated_names: list[str] = []
    seen_groups: set[str] = set()

    for name in feature_names:
        if name in member_to_group:
            group = member_to_group[name]
            if group in seen_groups:
                continue
            idxs = [name_to_idx[m] for m in group_definitions[group]]
            aggregated_shap.append(shap_values[:, idxs].sum(axis=1))
            aggregated_X.append(X[:, idxs].sum(axis=1))
            aggregated_names.append(group)
            seen_groups.add(group)
        else:
            idx = name_to_idx[name]
            aggregated_shap.append(shap_values[:, idx])
            aggregated_X.append(X[:, idx])
            aggregated_names.append(name)

    stacked_shap = np.column_stack(aggregated_shap)
    stacked_X = np.column_stack(aggregated_X)
    return stacked_shap, stacked_X, aggregated_names


def aggregate_ohe_arrays(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: Sequence[str],
    extra_groups: Mapping[str, Sequence[str]] | None = None,
    features_dict_path: Path = Path("data/features_dict.json"),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Public wrapper to aggregate OHE SHAP arrays with optional extra group definitions."""
    return _aggregate_ohe_vectors(
        shap_values=shap_values,
        X=X,
        feature_names=list(feature_names),
        features_dict_path=features_dict_path,
        extra_groups=extra_groups,
    )


def load_feature_group_map(
    feature_names: Sequence[str],
    groups_path: Union[str, Path],
) -> Dict[str, str]:
    """Build a feature->group map from a JSON spec with include/exclude rules."""
    feature_names = [str(f) for f in feature_names]
    data = json.loads(Path(groups_path).read_text())
    mapping: Dict[str, str] = {}

    def matches(name: str, group_spec: dict) -> bool:
        if "include_exact" in group_spec and name in group_spec["include_exact"]:
            return True
        if "include_prefix" in group_spec and any(
            name.startswith(p) for p in group_spec["include_prefix"]
        ):
            return True
        if "include_regex" in group_spec and any(
            re.search(pattern, name) for pattern in group_spec["include_regex"]
        ):
            return True
        return False

    def excluded(name: str, group_spec: dict) -> bool:
        if "exclude_exact" in group_spec and name in group_spec["exclude_exact"]:
            return True
        if "exclude_prefix" in group_spec and any(
            name.startswith(p) for p in group_spec["exclude_prefix"]
        ):
            return True
        if "exclude_regex" in group_spec and any(
            re.search(pattern, name) for pattern in group_spec["exclude_regex"]
        ):
            return True
        return False

    for group_name, spec in data.items():
        for name in feature_names:
            if matches(name, spec) and not excluded(name, spec):
                mapping[name] = group_name
    return mapping


def _resolve_sample_row_index(
    sample_identifier: int,
    sample_idx_map: np.ndarray | None,
    total_rows: int,
) -> int:
    """Map a design-matrix index to the corresponding row in a SHAP/survival array."""
    sample_identifier = int(sample_identifier)
    if sample_idx_map is not None:
        matches = np.where(sample_idx_map == sample_identifier)[0]
        if matches.size > 0:
            return int(matches[0])
    if 0 <= sample_identifier < total_rows:
        return sample_identifier
    raise IndexError(
        f"Sample index {sample_identifier} not found in SHAP payload "
        f"and exceeds available rows ({total_rows})."
    )


def _aggregate_shap_by_group(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    feature_groups: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    grouped_shap: list[np.ndarray] = []
    grouped_X: list[np.ndarray] = []
    grouped_names: list[str] = []
    for group_name, members in feature_groups.items():
        idxs = [name_to_idx[m] for m in members if m in name_to_idx]
        if not idxs:
            continue
        grouped_names.append(group_name)
        grouped_shap.append(shap_values[:, idxs].sum(axis=1))
        grouped_X.append(X[:, idxs].mean(axis=1))
    if not grouped_names:
        raise ValueError("No overlap between SHAP features and feature_groups.")
    return (
        np.column_stack(grouped_shap),
        np.column_stack(grouped_X),
        grouped_names,
    )


def prepare_grouped_shap(
    npz_path: Path,
    feature_groups: dict[str, list[str]],
    label_map: Dict[str, str],
) -> dict[str, Any]:
    shap_values, X, feature_names, base_value, _ = _load_shap_payload(npz_path)
    shap_values, X, feature_names = _aggregate_ohe_vectors(
        shap_values,
        X,
        feature_names,
    )
    grouped_shap, grouped_X, grouped_names = _aggregate_shap_by_group(
        shap_values=shap_values,
        X=X,
        feature_names=feature_names,
        feature_groups=feature_groups,
    )

    mean_abs = np.mean(np.abs(grouped_shap), axis=0)
    order = np.argsort(mean_abs)[::-1]
    grouped_shap = grouped_shap[:, order]
    grouped_X = grouped_X[:, order]
    grouped_names = [grouped_names[i] for i in order]
    display_names = [label_map.get(name, name) for name in grouped_names]

    sample_idx = int(np.abs(grouped_shap).sum(axis=1).argmax())
    return {
        "grouped_shap": grouped_shap,
        "grouped_X": grouped_X,
        "display_names": display_names,
        "base_value": base_value,
        "sample_idx": sample_idx,
        "run_label": npz_path.parent.name,
    }


# Separate plotting helpers for grouped SHAP
def plot_grouped_shap_summary_plot(
    payload: dict[str, Any],
    max_display: int,
    output_path: Path,
    xlim: tuple[float, float],
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        payload["grouped_shap"],
        payload["grouped_X"],
        feature_names=payload["display_names"],
        max_display=max_display,
        show=False,
    )
    plt.xlim(xlim)
    plt.tight_layout()
    plt.savefig(output_path, dpi=800)
    plt.close()
    return output_path


def plot_single_sample_shap_waterfall(
    npz_path: Path,
    sample_idx: int,
    title: str,
    label_map: Dict[str, str],
    max_display: int,
    figsize: tuple[float, float],
    dpi: int,
    shap_dir: Path,
) -> Path:
    """
    Render a SHAP waterfall plot for one sample using the raw SHAP output.
    """

    shap_values, X, feature_names, base_value, sample_idx_map = _load_shap_payload(
        npz_path
    )
    row_idx = _resolve_sample_row_index(
        sample_idx,
        sample_idx_map,
        shap_values.shape[0],
    )
    display_names = [label_map.get(name, name) for name in feature_names]
    explanation = shap.Explanation(
        values=shap_values[row_idx],
        base_values=base_value,
        data=X[row_idx],
        feature_names=display_names,
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)

    # append patient info to the figure title if available
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_title("Feature composition to risk")

    fig.set_size_inches(*figsize)
    output_path = shap_dir / f"{title}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def _ensure_png_path(output_path: Union[str, Path]) -> Path:
    path = Path(output_path)
    if path.suffix == "":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _assign_colors(
    keys: Sequence[str], palette: Sequence[str] = PROJECT_COLOR_CYCLE
) -> Dict[str, str]:
    seen: list[str] = []
    for key in keys:
        if key not in seen:
            seen.append(key)
    return {key: palette[i % len(palette)] for i, key in enumerate(seen)}


def _load_feature_importance_map(path: Path) -> Dict[str, float]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        if "importance" in raw:
            raw = raw["importance"]
        elif "items" in raw and isinstance(raw["items"], list):
            raw = raw["items"]
    if isinstance(raw, list):
        return {str(k): float(v) for k, v in raw}
    return {str(k): float(v) for k, v in raw.items()}


def _extract_scalar_metric(
    entry: Mapping[str, Any],
    metric: str,
) -> Tuple[Optional[float], Optional[float]]:
    block = entry.get(metric)
    if isinstance(block, Mapping):
        mean = block.get("mean")
        std = block.get("std")
    else:
        mean = block
        std = None

    def first(val: Any) -> Optional[float]:
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val, dtype=float)
            return float(arr[0]) if arr.size else None
        if val is None:
            return None
        return float(val)

    return first(mean), first(std)


def _parse_ablation_modality(group_label: str, prefix: str) -> str:
    base = Path(group_label).name
    if base.endswith("_aggregate.json"):
        base = base[: -len("_aggregate.json")]
    if base.endswith("_aggregate"):
        base = base[: -len("_aggregate")]
    if base.startswith(prefix):
        base = base[len(prefix) :]
    return base


def plot_km_hr_her2_with_risk_table(
    dm: pd.DataFrame,
    title: str,
    figsize: Optional[tuple[float, float]] = None,
    dpi: int = 400,
    output_path: Union[str, Path] = Path("km_hr_her2.png"),
) -> Tuple[Path, pd.DataFrame]:
    plt.rcParams.update(STYLE)

    def as_bool(x: Any) -> bool:
        if isinstance(x, str):
            return x.strip().lower() in {
                "1",
                "true",
                "t",
                "+",
                "pos",
                "positive",
                "yes",
                "y",
            }
        return bool(x)

    groups = list(dm.groupby(["HR", "HER2"]))
    groups.sort(key=lambda item: (as_bool(item[0][1]), as_bool(item[0][0])))
    labels = [
        f"HR{'+' if as_bool(hr) else '-'} / HER2{'+' if as_bool(her2) else '-'}"
        for (hr, her2), _ in groups
    ]
    palette = [
        PROJECT_COLOR_CYCLE[0],  # blue
        PROJECT_COLOR_CYCLE[1],  # orange
        PROJECT_COLOR_CYCLE[2],  # green
        PROJECT_COLOR_CYCLE[3],  # lighter blue
    ]
    colors = {label: palette[idx % len(palette)] for idx, label in enumerate(labels)}
    colors["Overall"] = "#333333"

    fig_size = SURVIVAL_FIGSIZE if figsize is None else figsize
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    kmfs: list[KaplanMeierFitter] = []
    for label, (_, g) in zip(labels, groups):
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=g["PFS_TIME_DAYS"],
            event_observed=g["PFS_EVENT"],
            label=label,
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=colors[label],
            lw=2.3,
            alpha=0.9,
            show_censors=False,
        )
        kmfs.append(kmf)

    kmf_all = KaplanMeierFitter()
    kmf_all.fit(
        durations=dm["PFS_TIME_DAYS"],
        event_observed=dm["PFS_EVENT"],
        label="Overall",
    )
    kmf_all.plot_survival_function(
        ax=ax,
        ci_show=False,
        color="#4E4E4E",
        lw=2.6,
        ls="--",
        alpha=0.95,
        show_censors=False,
    )

    ax.set_title(title, fontsize=SURVIVAL_TITLE_SIZE)
    ax.set_xlabel("Days since index", fontsize=SURVIVAL_LABEL_SIZE)
    ax.set_ylabel("Progression-free survival probability", fontsize=SURVIVAL_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=SURVIVAL_TICK_SIZE)
    ax.set_xlim(SURVIVAL_X_LIMIT)
    ax.set_ylim(SURVIVAL_Y_LIMIT)
    ax.grid(True, which="major", alpha=0.35, linestyle="--", linewidth=0.7)
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    timepoints_days = list(PFS_TIME_BIN_EDGES)
    risk_rows = []
    for label, (_, g) in zip(labels, groups):
        t = g["PFS_TIME_DAYS"].to_numpy()
        n0 = int((t >= 0).sum())
        at_risk = [int((t >= d).sum()) for d in timepoints_days]
        pct = [100.0 * n / n0 for n in at_risk]
        row = {"group": label}
        for d, n, p in zip(timepoints_days, at_risk, pct):
            row[f"{d}d_n"] = n
            row[f"{d}d_pct"] = p
        risk_rows.append(row)

    overall_row = {"group": "Overall"}
    t_all = dm["PFS_TIME_DAYS"].to_numpy()
    n0_all = int((t_all >= 0).sum())
    at_risk_all = [int((t_all >= d).sum()) for d in timepoints_days]
    pct_all = [100.0 * n / n0_all for n in at_risk_all]
    for d, n, p in zip(timepoints_days, at_risk_all, pct_all):
        overall_row[f"{d}d_n"] = n
        overall_row[f"{d}d_pct"] = p
    risk_rows.append(overall_row)

    risk_df = pd.DataFrame(risk_rows).set_index("group")
    return target, risk_df


def plot_km_late_start_vs_rest(
    dm: pd.DataFrame,
    late_start_indices: Sequence[int],
    output_path: Union[str, Path],
    title: str = "KM curves: rest vs late-start agents",
    figsize: tuple[float, float] = (7.5, 4.5),
    dpi: int = 300,
) -> Path:
    plt.rcParams.update(STYLE)
    late_start_idx = np.asarray(late_start_indices, dtype=int)
    all_idx = np.arange(dm.shape[0], dtype=int)
    rest_idx = np.setdiff1d(all_idx, late_start_idx)

    def _km_curve(indices: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
        rows = dm.iloc[indices]
        durations = pd.to_numeric(rows[TIME_COL], errors="coerce").to_numpy(dtype=float)
        events = pd.to_numeric(rows[EVENT_COL], errors="coerce").to_numpy(dtype=int)
        mask = np.isfinite(durations) & np.isfinite(events)
        if not np.any(mask):
            raise ValueError(f"No valid KM data for {label}.")
        kmf = KaplanMeierFitter()
        kmf.fit(durations=durations[mask], event_observed=events[mask], label=label)
        km_series = kmf.survival_function_[kmf._label]
        return km_series.index.to_numpy(dtype=float), km_series.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    idx_rest, vals_rest = _km_curve(rest_idx, "Rest")
    idx_late, vals_late = _km_curve(late_start_idx, "Late start")
    ax.step(idx_rest, vals_rest, where="post", linewidth=2.0, label="Rest")
    ax.step(idx_late, vals_late, where="post", linewidth=2.0, label="Late start")
    ax.set_xlabel("Days since index")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target


def plot_km_modeled_vs_no_rad_prior(
    modeled_df: pd.DataFrame,
    no_rad_prior_df: pd.DataFrame,
    time_col: str = TIME_COL,
    event_col: str = EVENT_COL,
    title: str = "Kaplan–Meier PFS: modeled vs no-radiology-within-90d-prior",
    figsize: tuple[float, float] = (7.5, 4.5),
    dpi: int = 300,
    output_path: Union[str, Path] = Path("km_modeled_vs_no_rad_prior.png"),
) -> Path:
    plt.rcParams.update(STYLE)
    d0 = pd.to_numeric(modeled_df[time_col], errors="coerce").to_numpy(dtype=float)
    e0 = pd.to_numeric(modeled_df[event_col], errors="coerce").to_numpy(dtype=int)
    m0 = np.isfinite(d0) & np.isfinite(e0)
    d0, e0 = d0[m0], e0[m0]

    d1 = pd.to_numeric(
        no_rad_prior_df[f"ORIGINAL_{time_col}"], errors="coerce"
    ).to_numpy(dtype=float)
    e1 = pd.to_numeric(
        no_rad_prior_df[f"ORIGINAL_{event_col}"], errors="coerce"
    ).to_numpy(dtype=int)
    m1 = np.isfinite(d1) & np.isfinite(e1)
    d1, e1 = d1[m1], e1[m1]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    kmf0 = KaplanMeierFitter()
    kmf0.fit(durations=d0, event_observed=e0, label="Modeled cohort")
    kmf0.plot_survival_function(
        ax=ax, ci_show=False, color=PROJECT_COLOR_CYCLE[0], lw=2.6
    )

    kmf1 = KaplanMeierFitter()
    kmf1.fit(durations=d1, event_observed=e1, label="Dropped: no radiology ≤90d prior")
    kmf1.plot_survival_function(
        ax=ax, ci_show=False, color=PROJECT_COLOR_CYCLE[1], lw=2.6, ls="--"
    )

    ax.set_title(title)
    ax.set_xlabel("Days since index")
    ax.set_ylabel("Progression-free survival probability")
    ax.set_xlim(SURVIVAL_X_LIMIT)
    ax.set_ylim(SURVIVAL_Y_LIMIT)
    ax.grid(True, which="major", alpha=0.35, linestyle="--", linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_mean_surv_late_start(
    dm: pd.DataFrame,
    time_grid: Sequence[float],
    surv_clean: pd.DataFrame,
    surv_leakage: pd.DataFrame,
    late_start_indices: Sequence[int],
    output_path: Union[str, Path],
    title: str = "Mean survival prediction: rest vs late-start agents",
    figsize: tuple[float, float] = (7.5, 4.5),
    dpi: int = 300,
) -> tuple[Path, Dict[str, float]]:
    plt.rcParams.update(STYLE)
    time_grid = np.asarray(time_grid, dtype=float).reshape(-1)
    late_start_idx = np.asarray(late_start_indices, dtype=int)
    all_idx = np.arange(surv_clean.shape[0], dtype=int)
    rest_idx = np.setdiff1d(all_idx, late_start_idx)

    # Colors: blues for clean, oranges for leakage, greens for KM.
    c_rest_clean = PROJECT_COLOR_CYCLE[0]
    c_late_clean = PROJECT_COLOR_CYCLE[3]
    c_rest_leak = PROJECT_COLOR_CYCLE[1]
    c_late_leak = PROJECT_COLOR_CYCLE[4]
    c_km_rest = PROJECT_COLOR_CYCLE[2]
    c_km_late = PROJECT_COLOR_CYCLE[5]

    def _mean_surv_curve(surv_df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
        subset = surv_df.iloc[indices].to_numpy(dtype=float)
        return np.nanmean(subset, axis=0)

    rest_clean = _mean_surv_curve(surv_clean, rest_idx)
    rest_leakage = _mean_surv_curve(surv_leakage, rest_idx)
    late_clean = _mean_surv_curve(surv_clean, late_start_idx)
    late_leakage = _mean_surv_curve(surv_leakage, late_start_idx)

    def _curve_mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.nanmean(np.abs(a - b)))

    def _km_curve(indices: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
        rows = dm.iloc[indices]
        durations = pd.to_numeric(rows[TIME_COL], errors="coerce").to_numpy(dtype=float)
        events = pd.to_numeric(rows[EVENT_COL], errors="coerce").to_numpy(dtype=int)
        mask = np.isfinite(durations) & np.isfinite(events)
        if not np.any(mask):
            raise ValueError(f"No valid KM data for {label}.")
        kmf = KaplanMeierFitter()
        kmf.fit(durations=durations[mask], event_observed=events[mask], label=label)
        km_series = kmf.survival_function_[kmf._label]
        return km_series.index.to_numpy(dtype=float), km_series.to_numpy(dtype=float)

    km_time_rest, km_vals_rest = _km_curve(rest_idx, "Rest (KM)")
    km_time_late, km_vals_late = _km_curve(late_start_idx, "Late start (KM)")

    rest_clean_km = np.interp(km_time_rest, time_grid, rest_clean)
    rest_leakage_km = np.interp(km_time_rest, time_grid, rest_leakage)
    late_clean_km = np.interp(km_time_late, time_grid, late_clean)
    late_leakage_km = np.interp(km_time_late, time_grid, late_leakage)

    mae = {
        "rest_clean_vs_km": _curve_mae(rest_clean_km, km_vals_rest),
        "rest_leakage_vs_km": _curve_mae(rest_leakage_km, km_vals_rest),
        "late_clean_vs_km": _curve_mae(late_clean_km, km_vals_late),
        "late_leakage_vs_km": _curve_mae(late_leakage_km, km_vals_late),
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(
        time_grid,
        rest_clean,
        linewidth=2.4,
        color=c_rest_clean,
        label="Baseline prediction for rest of cohort",
    )
    ax.plot(
        time_grid,
        late_clean,
        linewidth=2.4,
        color=c_late_clean,
        label="Baseline prediction for lines with late-start agents",
    )
    ax.plot(
        time_grid,
        rest_leakage,
        linewidth=2.4,
        color=c_rest_leak,
        label="Leakage model prediction for rest of cohort",
    )
    ax.plot(
        time_grid,
        late_leakage,
        linewidth=2.4,
        color=c_late_leak,
        label="Leakage model prediction for lines with late-start agents",
    )
    ax.step(
        km_time_rest,
        km_vals_rest,
        where="post",
        linewidth=2.2,
        color=c_km_rest,
        ls="--",
        label="KM curve for rest of cohort",
    )
    ax.step(
        km_time_late,
        km_vals_late,
        where="post",
        linewidth=2.2,
        color=c_km_late,
        ls="--",
        label="KM curve for lines with late-start agents",
    )
    ax.set_xlabel("Days since index")
    ax.set_ylabel("Mean predicted survival")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target, mae


def plot_mean_surv_late_start_systemic_vs_local(
    dm: pd.DataFrame,
    time_grid: Sequence[float],
    surv_df: pd.DataFrame,
    late_start_systemic_indices: Sequence[int],
    late_start_local_indices: Sequence[int],
    output_path: Union[str, Path],
    *,
    title: str = "Mean survival vs KM: late-start systemic vs local",
    mean_colors: tuple[Any, Any] | None = None,
    km_colors: tuple[Any, Any] | None = None,
    figsize: tuple[float, float] = (7.5, 4.5),
    dpi: int = 300,
) -> tuple[Path, Dict[str, float]]:
    plt.rcParams.update(STYLE)
    time_grid = np.asarray(time_grid, dtype=float).reshape(-1)
    systemic_idx = np.asarray(late_start_systemic_indices, dtype=int)
    local_idx = np.asarray(late_start_local_indices, dtype=int)

    if systemic_idx.size == 0 or local_idx.size == 0:
        raise ValueError("Both systemic and local indices must be non-empty.")

    if mean_colors is None:
        mean_colors = (PROJECT_COLOR_CYCLE[0], PROJECT_COLOR_CYCLE[3])
    if km_colors is None:
        km_colors = (PROJECT_COLOR_CYCLE[2], PROJECT_COLOR_CYCLE[5])

    def _mean_surv_curve(indices: np.ndarray) -> np.ndarray:
        subset = surv_df.iloc[indices].to_numpy(dtype=float)
        return np.nanmean(subset, axis=0)

    def _curve_mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.nanmean(np.abs(a - b)))

    def _km_curve(indices: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
        rows = dm.iloc[indices]
        durations = pd.to_numeric(rows[TIME_COL], errors="coerce").to_numpy(dtype=float)
        events = pd.to_numeric(rows[EVENT_COL], errors="coerce").to_numpy(dtype=int)
        mask = np.isfinite(durations) & np.isfinite(events)
        if not np.any(mask):
            raise ValueError(f"No valid KM data for {label}.")
        kmf = KaplanMeierFitter()
        kmf.fit(durations=durations[mask], event_observed=events[mask], label=label)
        km_series = kmf.survival_function_[kmf._label]
        return km_series.index.to_numpy(dtype=float), km_series.to_numpy(dtype=float)

    systemic_mean = _mean_surv_curve(systemic_idx)
    local_mean = _mean_surv_curve(local_idx)

    km_time_systemic, km_vals_systemic = _km_curve(
        systemic_idx, "Late-start systemic (KM)"
    )
    km_time_local, km_vals_local = _km_curve(local_idx, "Late-start local (KM)")

    systemic_mean_km = np.interp(km_time_systemic, time_grid, systemic_mean)
    local_mean_km = np.interp(km_time_local, time_grid, local_mean)

    mae = {
        "systemic_vs_km": _curve_mae(systemic_mean_km, km_vals_systemic),
        "local_vs_km": _curve_mae(local_mean_km, km_vals_local),
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(
        time_grid,
        systemic_mean,
        linewidth=2.6,
        color=mean_colors[0],
        label="Late-start systemic (mean pred)",
    )
    ax.plot(
        time_grid,
        local_mean,
        linewidth=2.6,
        color=mean_colors[1],
        label="Late-start local (mean pred)",
    )
    ax.step(
        km_time_systemic,
        km_vals_systemic,
        where="post",
        linewidth=2.2,
        color=km_colors[0],
        ls="--",
        label="Late-start systemic (KM)",
    )
    ax.step(
        km_time_local,
        km_vals_local,
        where="post",
        linewidth=2.2,
        color=km_colors[1],
        ls="--",
        label="Late-start local (KM)",
    )

    ax.set_xlabel("Days since index")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target, mae


def plot_mean_surv_late_start_systemic_vs_local_clean_leakage(
    dm: pd.DataFrame,
    time_grid: Sequence[float],
    surv_clean: pd.DataFrame,
    surv_leakage: pd.DataFrame,
    late_start_systemic_indices: Sequence[int],
    late_start_local_indices: Sequence[int],
    output_path: Union[str, Path],
    *,
    title: str = "Mean survival vs KM: systemic vs local late-start (clean + leakage)",
    figsize: tuple[float, float] = (7.8, 4.8),
    dpi: int = 300,
) -> tuple[Path, Dict[str, float]]:
    plt.rcParams.update(STYLE)
    time_grid = np.asarray(time_grid, dtype=float).reshape(-1)
    systemic_idx = np.asarray(late_start_systemic_indices, dtype=int)
    local_idx = np.asarray(late_start_local_indices, dtype=int)

    if systemic_idx.size == 0 or local_idx.size == 0:
        raise ValueError("Both systemic and local indices must be non-empty.")

    c_systemic_clean = PROJECT_COLOR_CYCLE[0]
    c_local_clean = PROJECT_COLOR_CYCLE[3]
    c_systemic_leak = PROJECT_COLOR_CYCLE[1]
    c_local_leak = PROJECT_COLOR_CYCLE[4]
    c_km_systemic = PROJECT_COLOR_CYCLE[2]
    c_km_local = PROJECT_COLOR_CYCLE[5]

    def _mean_surv_curve(surv_df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
        subset = surv_df.iloc[indices].to_numpy(dtype=float)
        return np.nanmean(subset, axis=0)

    def _curve_mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.nanmean(np.abs(a - b)))

    def _km_curve(indices: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
        rows = dm.iloc[indices]
        durations = pd.to_numeric(rows[TIME_COL], errors="coerce").to_numpy(dtype=float)
        events = pd.to_numeric(rows[EVENT_COL], errors="coerce").to_numpy(dtype=int)
        mask = np.isfinite(durations) & np.isfinite(events)
        if not np.any(mask):
            raise ValueError(f"No valid KM data for {label}.")
        kmf = KaplanMeierFitter()
        kmf.fit(durations=durations[mask], event_observed=events[mask], label=label)
        km_series = kmf.survival_function_[kmf._label]
        return km_series.index.to_numpy(dtype=float), km_series.to_numpy(dtype=float)

    systemic_clean = _mean_surv_curve(surv_clean, systemic_idx)
    local_clean = _mean_surv_curve(surv_clean, local_idx)
    systemic_leak = _mean_surv_curve(surv_leakage, systemic_idx)
    local_leak = _mean_surv_curve(surv_leakage, local_idx)

    km_time_systemic, km_vals_systemic = _km_curve(systemic_idx, "Systemic (KM)")
    km_time_local, km_vals_local = _km_curve(local_idx, "Local (KM)")

    systemic_clean_km = np.interp(km_time_systemic, time_grid, systemic_clean)
    systemic_leak_km = np.interp(km_time_systemic, time_grid, systemic_leak)
    local_clean_km = np.interp(km_time_local, time_grid, local_clean)
    local_leak_km = np.interp(km_time_local, time_grid, local_leak)

    mae = {
        "systemic_clean_vs_km": _curve_mae(systemic_clean_km, km_vals_systemic),
        "systemic_leakage_vs_km": _curve_mae(systemic_leak_km, km_vals_systemic),
        "local_clean_vs_km": _curve_mae(local_clean_km, km_vals_local),
        "local_leakage_vs_km": _curve_mae(local_leak_km, km_vals_local),
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(
        time_grid,
        systemic_clean,
        linewidth=2.4,
        color=c_systemic_clean,
        label="Systemic (clean mean)",
    )
    ax.plot(
        time_grid,
        local_clean,
        linewidth=2.4,
        color=c_local_clean,
        label="Local (clean mean)",
    )
    ax.plot(
        time_grid,
        systemic_leak,
        linewidth=2.4,
        color=c_systemic_leak,
        label="Systemic (leakage mean)",
    )
    ax.plot(
        time_grid,
        local_leak,
        linewidth=2.4,
        color=c_local_leak,
        label="Local (leakage mean)",
    )
    ax.step(
        km_time_systemic,
        km_vals_systemic,
        where="post",
        linewidth=2.2,
        color=c_km_systemic,
        ls="--",
        label="Systemic (KM)",
    )
    ax.step(
        km_time_local,
        km_vals_local,
        where="post",
        linewidth=2.2,
        color=c_km_local,
        ls="--",
        label="Local (KM)",
    )

    ax.set_xlabel("Days since index")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target, mae


def _sample_series(
    entry: Mapping[str, Any],
    time_field: str,
    metric: str,
    time_points: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(entry[time_field]["mean"], dtype=float)
    values = np.asarray(entry[metric]["mean"], dtype=float)
    std = np.asarray(entry[metric].get("std", np.zeros_like(values)), dtype=float)
    xs = []
    mu = []
    sd = []
    for target in time_points:
        idx = int(np.argmin(np.abs(times - float(target))))
        xs.append(float(target))
        mu.append(float(values[idx]))
        sd.append(float(std[idx]) if std.size else 0.0)
    return np.asarray(xs), np.asarray(mu), np.asarray(sd)


def _align_series(
    entry: Mapping[str, Any],
    metric: str,
    time_points: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Normalize mean/std series from flexible dict formats and sample if needed."""
    # dataclass-like support
    if hasattr(entry, "times") and hasattr(entry, "mean") and hasattr(entry, "std"):
        times = np.asarray(entry.times, dtype=float).reshape(-1)
        mean = np.asarray(entry.mean, dtype=float).reshape(-1)
        std = np.asarray(entry.std, dtype=float).reshape(mean.shape)
        runs = int(getattr(entry, "n", getattr(entry, "num_runs", 1)))
    else:
        time_raw = entry.get("time", entry.get("time_grid", entry.get("eval_times")))
        if time_raw is None:
            raise ValueError("Series entry missing time grid.")
        if isinstance(time_raw, Mapping):
            time_raw = time_raw.get("mean", time_raw)
        times = np.asarray(time_raw, dtype=float).reshape(-1)

        metric_entry = entry.get(metric, None)
        if metric_entry is None:
            raise ValueError(f"Series entry missing metric '{metric}'.")
        if isinstance(metric_entry, Mapping):
            mean = metric_entry.get("mean", metric_entry)
            std = metric_entry.get("std", np.zeros_like(mean))
            runs = int(metric_entry.get("n", entry.get("num_runs", 1)))
        else:
            mean = metric_entry
            std = np.zeros_like(mean)
            runs = int(entry.get("num_runs", 1))
        mean = np.asarray(mean, dtype=float).reshape(-1)
        std = np.asarray(std, dtype=float).reshape(mean.shape)

    order = np.argsort(times)
    times = times[order]
    mean = mean[order]
    std = std[order]

    drop_mask = ~(np.isclose(times, 360.0) | np.isclose(times, 720.0))
    times = times[drop_mask]
    mean = mean[drop_mask]
    std = std[drop_mask]

    if time_points is None:
        xs, mu, sd = times, mean, std
    else:
        time_points_arr = np.asarray(time_points, dtype=float).reshape(-1)
        time_points_arr = time_points_arr[
            ~(np.isclose(time_points_arr, 360.0) | np.isclose(time_points_arr, 720.0))
        ]
        time_points_arr = np.sort(time_points_arr)
        xs, mu, sd = _sample_series(
            {"time": {"mean": times}, metric: {"mean": mean, "std": std}},
            "time",
            metric,
            time_points_arr,
        )
    return xs, mu, sd, runs


def plot_full_group_individual_time_curves(
    models: Sequence[str],
    metric: str,
    time_points: Sequence[float],
    max_time_days: float,
    shade: str,
    shade_alpha: float,
    y_min_auc_ipcw: float,
    y_limits: Mapping[str, Tuple[float, float]],
    title: str,
    figsize: tuple[float, float],
    dpi: int,
    output_path: Union[str, Path],
    series: Mapping[str, Mapping[str, Any]],
    model_name_map: Mapping[str, str] | None = None,
) -> Path:
    plt.rcParams.update(STYLE)

    if series is None or not series:
        raise ValueError("series must be provided for plotting.")
    if models is None or not models:
        raise ValueError("models must be provided for plotting.")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    colors = _assign_colors(models, palette=PROJECT_COLOR_CYCLE)

    lo_vals: list[float] = []
    hi_vals: list[float] = []

    for name in models:
        entry = series[name]
        xs, mu, sd, runs = _align_series(entry, metric, time_points)
        if max_time_days is not None:
            mask = xs <= float(max_time_days)
            xs, mu, sd = xs[mask], mu[mask], sd[mask]
        runs = max(int(runs), 1)
        half = 1.96 * sd / np.sqrt(float(runs)) if shade == "ci" and runs > 1 else sd
        (line,) = ax.plot(
            xs,
            mu,
            marker="o",
            linewidth=2.0,
            label=model_name_map.get(name, name) if model_name_map else name,
            color=colors[name],
        )
        ax.fill_between(xs, mu - half, mu + half, color=colors[name], alpha=shade_alpha)
        lo_vals.extend((mu - half).tolist())
        hi_vals.extend((mu + half).tolist())

    vmin = float(np.min(lo_vals))
    vmax = float(np.max(hi_vals))
    span = max(vmax - vmin, 1e-6)
    lo = vmin - 0.05 * span
    hi = vmax + 0.05 * span
    if metric in {"ipcw", "auc"}:
        lo = max(0.0, max(lo, float(y_min_auc_ipcw)))
        hi = min(1.0, hi)
    if y_limits and metric in y_limits:
        lo, hi = y_limits[metric]

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(metric)
    ax.set_ylim(lo, hi)
    if time_points is not None:
        ax.set_xticks(list(time_points))
    ax.grid(True, color="#cccccc", linewidth=0.6, alpha=0.4, linestyle="--")
    ax.set_title(title)
    ax.legend(loc="upper center", ncols=2)
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_hrher2_time_curves(
    model: str,
    metric: str,
    time_points: Sequence[float],
    include_overall: bool,
    shade: str,
    shade_alpha: float,
    y_limits: Tuple[float, float],
    title: str,
    figsize: tuple[float, float],
    dpi: int,
    output_path: Union[str, Path],
    series_overall: Mapping[str, Any],
    subgroup_series: Mapping[str, Mapping[str, Any]],
    model_name_map: Mapping[str, str] | None = None,
) -> Path:
    plt.rcParams.update(STYLE)

    entry = series_overall
    if entry is None:
        raise ValueError("series_overall is required for HR/HER2 plots.")
    if subgroup_series is None:
        raise ValueError("subgroup_series is required for HR/HER2 plots.")
    display_model = model_name_map.get(model, model) if model_name_map else model
    if title and model in title:
        title = title.replace(model, display_model)
    elif title and "{model}" in title:
        title = title.format(model=display_model)
    curves = [
        ("HR-/HER2-", "HR-_HER2-"),
        ("HR-/HER2+", "HR-_HER2+"),
        ("HR+/HER2-", "HR+_HER2-"),
        ("HR+/HER2+", "HR+_HER2+"),
    ]
    labels = [label for label, _ in curves]
    if include_overall:
        labels = ["Full cohort"] + labels
    colors = _assign_colors(labels, palette=PROJECT_COLOR_CYCLE)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    lo_vals: list[float] = []
    hi_vals: list[float] = []

    if include_overall:
        xs, mu, sd, runs = _align_series(entry, metric, time_points)
        runs = max(int(runs), 1)
        half = 1.96 * sd / np.sqrt(float(runs)) if shade == "ci" and runs > 1 else sd
        ax.plot(xs, mu, marker="o", label="Full cohort", color=colors["Full cohort"])
        ax.fill_between(
            xs,
            mu - half,
            mu + half,
            color=colors["Full cohort"],
            alpha=shade_alpha,
        )
        lo_vals.extend((mu - half).tolist())
        hi_vals.extend((mu + half).tolist())

    for label, key in curves:
        if subgroup_series is None or key not in subgroup_series:
            continue
        node = subgroup_series[key]
        xs, mu, sd, runs = _align_series(node, metric, time_points)
        runs = max(int(getattr(node, "n", getattr(node, "num_runs", runs))), 1)
        half = 1.96 * sd / np.sqrt(float(runs)) if shade == "ci" and runs > 1 else sd
        ax.plot(xs, mu, marker="o", label=label, color=colors[label])
        ax.fill_between(
            xs, mu - half, mu + half, color=colors[label], alpha=shade_alpha
        )
        lo_vals.extend((mu - half).tolist())
        hi_vals.extend((mu + half).tolist())

    vmin = float(np.min(lo_vals))
    vmax = float(np.max(hi_vals))
    span = max(vmax - vmin, 1e-6)
    lo = vmin - 0.05 * span
    hi = vmax + 0.05 * span
    if metric in {"ipcw", "auc"}:
        lo = max(0.0, lo)
        hi = min(1.0, hi)
    lo, hi = y_limits

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(metric)
    ax.set_ylim(lo, hi)
    if time_points is not None:
        ax.set_xticks(list(time_points))
    ax.grid(True, color="#cccccc", linewidth=0.6, alpha=0.4, linestyle="--")
    ax.set_title(title)
    ax.legend(loc="upper center", ncols=2)
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_lot_time_curves(
    aggregate_path: Union[str, Path],
    model: str,
    metric: str,
    time_points: Sequence[float],
    shade: str,
    shade_alpha: float,
    y_limits: Tuple[float, float],
    title: str,
    figsize: tuple[float, float],
    dpi: int,
    output_path: Union[str, Path],
) -> Path:
    plt.rcParams.update(STYLE)

    entry = json.loads(Path(aggregate_path).read_text())["model_statistics"][model]
    lot_metrics = entry.get("lot_metrics", {})
    curves = [
        ("LoT1", "LINE_1"),
        ("LoT 2+", "LINE_2_PLUS"),
    ]
    labels = [label for label, key in curves if key in lot_metrics]
    if not labels:
        raise ValueError(f"{model} missing lot_metrics in {aggregate_path}")
    colors = _assign_colors(labels, palette=PROJECT_COLOR_CYCLE)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    lo_vals: list[float] = []
    hi_vals: list[float] = []

    for label, key in curves:
        if key not in lot_metrics:
            continue
        node = lot_metrics[key]
        xs, mu, sd = _sample_series(node, "eval_times", metric, time_points)
        runs = max(int(node.get("num_runs", 1)), 1)
        half = 1.96 * sd / np.sqrt(float(runs)) if shade == "ci" and runs > 1 else sd
        ax.plot(xs, mu, marker="o", label=label, color=colors[label])
        ax.fill_between(
            xs,
            mu - half,
            mu + half,
            color=colors[label],
            alpha=shade_alpha,
        )
        lo_vals.extend((mu - half).tolist())
        hi_vals.extend((mu + half).tolist())

    vmin = float(np.min(lo_vals))
    vmax = float(np.max(hi_vals))
    span = max(vmax - vmin, 1e-6)
    lo = vmin - 0.05 * span
    hi = vmax + 0.05 * span
    if metric in {"ipcw", "auc"}:
        lo = max(0.0, lo)
        hi = min(1.0, hi)
    lo, hi = y_limits

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(metric)
    ax.set_ylim(lo, hi)
    ax.set_xticks(list(time_points))
    ax.grid(True, color="#cccccc", linewidth=0.6, alpha=0.4, linestyle="--")
    ax.set_title(title)
    ax.legend(loc="upper center", ncols=2)
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def _collect_ablation_values(
    aggregate_paths: Mapping[str, Union[str, Path]],
    prefix: str,
    models: Sequence[str],
    metric: str,
) -> Tuple[
    list[str],
    Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]],
    Dict[str, Optional[float]],
]:
    stats_by_group = {
        group: json.loads(Path(path).read_text())["model_statistics"]
        for group, path in aggregate_paths.items()
    }
    baseline = stats_by_group["full"]
    modalities = sorted(
        _parse_ablation_modality(group, prefix)
        for group in stats_by_group
        if group != "full" and group.startswith(prefix)
    )
    values: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
    for modality in modalities:
        values[modality] = {}
        group_key = f"{prefix}{modality}"
        group_stats = stats_by_group[group_key]
        for model in models:
            try:
                values[modality][model] = _extract_scalar_metric(
                    group_stats[model], metric
                )
            except KeyError:
                continue
    baseline_values = {
        model: _extract_scalar_metric(baseline[model], metric)[0] for model in models
    }
    return modalities, values, baseline_values


def plot_ablation_metric_bars_grid(
    aggregate_paths: Mapping[str, Union[str, Path]],
    prefix: str,
    models: Sequence[str],
    metric: str,
    title: str,
    figsize: tuple[float, float],
    dpi: int,
    output_path: Union[str, Path],
) -> Path:
    plt.rcParams.update(STYLE)

    modalities, values_map, baseline_values = _collect_ablation_values(
        aggregate_paths, prefix, models, metric
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    model_colors = _assign_colors(models)
    width = 0.8 / max(len(models), 1)
    positions = np.arange(len(modalities), dtype=float)
    labels = [mod for mod in modalities]

    extrema: list[float] = []

    for idx, model in enumerate(models):
        heights = []
        errs = []
        for mod in modalities:
            try:
                mean, std = values_map[mod][model]
            except KeyError:
                mean, std = None, None
            base = baseline_values[model]
            delta = (mean - base) if mean is not None and base is not None else 0.0
            heights.append(delta)
            errs.append(float(std or 0.0))
            extrema.extend([delta - errs[-1], delta + errs[-1]])

        offsets = (idx - (len(models) - 1) / 2.0) * width
        ax.bar(
            positions + offsets,
            heights,
            width=width,
            color=model_colors[model],
            alpha=0.9,
            label=model,
        )
        ax.errorbar(
            positions + offsets,
            heights,
            yerr=errs,
            fmt="none",
            ecolor=model_colors[model],
            elinewidth=0.8,
            capsize=2.5,
            linestyle=":",
            alpha=0.85,
        )

    limit = max(abs(val) for val in extrema) if extrema else 0.05
    pad = max(0.01, 0.12 * limit)
    ax.set_ylim(-limit - pad, limit + pad)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.8)
    ax.grid(True, axis="y", color="#bfbfbf", linewidth=0.5, alpha=0.45, linestyle="--")
    ax.set_ylabel(f"Δ{metric}")
    ax.set_title(title)
    ax.legend(loc="upper center", ncols=len(models))
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_modality_best_model_metric_deltas(
    modality_perf: Mapping,
    baseline_stat,
    ablation_type: str,
    output_path: Path,
    figsize: tuple[float, float],
    dpi: int,
    model_name_map: Mapping[str, str] | None = None,
) -> Path:
    metrics = ["C", "mean_auc"]
    metric_labels = {"C": "C-index", "mean_auc": "Mean AUC"}

    modalities = sorted(modality_perf.keys())
    positions = np.arange(len(modalities), dtype=float)
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    extremes = []
    for j, metric in enumerate(metrics):
        heights = []
        errs = []
        offsets = positions + (j - (len(metrics) - 1) / 2.0) * width

        baseline_mean = float(baseline_stat.scalars[metric].mean)

        for modality in modalities:
            entry = modality_perf[modality][ablation_type]
            best_model = entry["best_model"]
            stats = entry[best_model]
            scalar = stats.scalars[metric]

            delta = float(scalar.mean) - baseline_mean
            heights.append(delta)
            errs.append(float(scalar.std))
            extremes.extend([delta - errs[-1], delta + errs[-1]])

        ax.bar(
            offsets, heights, width=width * 0.9, alpha=0.9, label=metric_labels[metric]
        )
        ax.errorbar(offsets, heights, yerr=errs, fmt="none", capsize=2.5, alpha=0.9)

    limit = max(abs(v) for v in extremes) if extremes else 0.05
    pad = max(0.01, 0.15 * limit)
    ax.set_ylim(-limit - pad, limit + pad)

    ax.axhline(0.0, linewidth=0.8, linestyle="--", alpha=0.8)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.4, linestyle="--")

    tick_labels = []
    for m in modalities:
        best_model = modality_perf[m][ablation_type]["best_model"]
        display_model = (
            model_name_map.get(best_model, best_model) if model_name_map else best_model
        )
        tick_labels.append(f"{m}\n({display_model})")
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=22, ha="right")

    ax.set_ylabel("Δ vs baseline (full model)")
    ax.set_title(f"{ablation_type.capitalize()} modality ablation (Δ)")
    ax.legend(loc="upper center", ncols=len(metrics))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_shap_summary_from_arrays(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: Sequence[str],
    label_map: Mapping[str, str] | None,
    top_k: int,
    title: str,
    figsize: tuple[float, float],
    dpi: int,
    output_path: Union[str, Path],
) -> Path:
    """Plot SHAP summary using in-memory arrays."""
    plt.rcParams.update(STYLE)
    shap_values = np.asarray(shap_values, dtype=float)
    X = np.asarray(X, dtype=float)
    if shap_values.shape != X.shape:
        raise ValueError("shap_values and X must have identical shapes.")
    display_names = [
        label_map.get(name, name) if label_map else name for name in feature_names
    ]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.sca(ax)
    # SHAP summary (beeswarm) uses random jitter for point placement; fix RNG for determinism.
    rng_state = np.random.get_state()
    np.random.seed(int(RANDOM_STATE))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=display_names,
        max_display=top_k,
        show=False,
        plot_size=None,
    )
    np.random.set_state(rng_state)
    ax.set_title(title)
    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target


def plot_single_sample_shap_waterfall_from_arrays(
    shap_values: np.ndarray,
    X: np.ndarray,
    expected_value: float,
    feature_names: Sequence[str],
    label_map: Mapping[str, str],
    title: str,
    max_display: int,
    figsize: tuple[float, float],
    dpi: int,
    output_path: Union[str, Path],
) -> Path:
    """Plot a waterfall for one sample using in-memory SHAP arrays."""
    plt.rcParams.update(STYLE)
    shap_values = np.asarray(shap_values, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float).reshape(-1)
    expected_value = float(np.asarray(expected_value).reshape(-1)[0])
    display_names = [label_map.get(name, name) for name in feature_names]
    explanation = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=X,
        feature_names=display_names,
    )
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    shap.plots.waterfall(
        explanation,
        max_display=max_display,
        show=False,
    )
    ax.set_title(title)
    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_group_mean_abs_shap(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    group_map: Mapping[str, str],
    label_map: Mapping[str, str] | None,
    top_k: int,
    title: str,
    figsize: tuple[float, float],
    dpi: int,
    output_path: Union[str, Path],
) -> Path:
    """Plot mean |SHAP| aggregated by a provided feature->group mapping.

    Aggregation is performed by summing SHAP values within each group for each sample,
    taking the absolute value, then averaging across samples.

    Parameters
    ----------
    normalize:
        Optional normalization for group magnitudes:
        - "none": plot mean(|sum_group_shap|) (default)
        - "per_feature": divide by number of features in the group
        - "fraction": for each sample, divide |sum_group_shap| by total
          |sum_group_shap| across groups; then average across samples
    """
    plt.rcParams.update(STYLE)
    shap_values = np.asarray(shap_values, dtype=float)
    feat_names = [str(f) for f in feature_names]
    if shap_values.ndim != 2:
        raise ValueError(
            f"Expected shap_values to be 2D (n_samples, n_features); got shape {shap_values.shape}."
        )
    if shap_values.shape[1] != len(feat_names):
        raise ValueError(
            "feature_names must align with shap_values columns: "
            f"{len(feat_names)} names vs {shap_values.shape[1]} SHAP columns."
        )

    group_to_indices: Dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(feat_names):
        group = group_map.get(name, "Other")
        if group == "Other":
            raise ValueError(f"Feature '{name}' missing from group_map.")
        group_to_indices[group].append(idx)

    group_mean: Dict[str, float] = {}
    n_samples = shap_values.shape[0]
    grouped_abs = np.zeros((n_samples, len(group_to_indices)), dtype=float)
    group_keys = list(group_to_indices.keys())

    for group_idx, group in enumerate(group_keys):
        indices = group_to_indices[group]
        group_sum = np.nansum(shap_values[:, indices], axis=1)
        grouped_abs[:, group_idx] = np.abs(group_sum)

    totals = np.nansum(grouped_abs, axis=1)
    # Compute per-sample contribution share to model output shift.
    shares = np.divide(
        grouped_abs,
        totals[:, None],
        out=np.zeros_like(grouped_abs),
        where=totals[:, None] > 0,
    )
    for group_idx, group in enumerate(group_keys):
        group_mean[group] = float(np.nanmean(shares[:, group_idx]))

    top = sorted(group_mean.items(), key=lambda x: x[1], reverse=True)[: int(top_k)]
    group_keys = [g for g, _ in top]
    labels = [label_map.get(g, g) if label_map else g for g in group_keys]
    values = [v for _, v in top]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    category_colors = {
        "Surveillance pattern": PROJECT_COLOR_CYCLE[0],
        "Biological intensity": PROJECT_COLOR_CYCLE[1],
        "Treatment": PROJECT_COLOR_CYCLE[2],
    }
    other_color = "#666666"
    pattern_keywords = {
        kw.lower()
        for kw in [
            "Measurement cadence",
            "Imaging coverage and cancer status",
        ]
    }
    bio_keywords = {
        kw.lower()
        for kw in [
            "Tumor marker kinetics",
            "Recent ECOG status",
            "Tumor site burden",
            "Genomics and therapeutic targets",
            "Initial diagnosis",
        ]
    }
    treatment_keywords = {
        kw.lower()
        for kw in [
            "Treatment exposure",
        ]
    }
    colors: list[str] = []
    categories: list[str] = []
    for key in group_keys:
        lower = key.lower()
        if any(kw in lower for kw in pattern_keywords):
            category = "Surveillance pattern"
        elif any(kw in lower for kw in bio_keywords):
            category = "Biological intensity"
        elif any(kw in lower for kw in treatment_keywords):
            category = "Treatment"
        else:
            print(f"Warning: unclassified group '{key}' for color assignment.")
            category = "Other"
        categories.append(category)
        colors.append(category_colors.get(category, other_color))

    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    xlabel = "Mean |Σ SHAP|"
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    legend_handles = []
    seen: set[str] = set()
    for category in ["Surveillance pattern", "Treatment", "Biological intensity"]:
        if category in categories and category not in seen:
            seen.add(category)
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=category_colors[category],
                    lw=6,
                    label=category.capitalize(),
                )
            )
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, loc="lower right")

    fig.tight_layout()
    target = _ensure_png_path(output_path)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


# === Risk stratification helpers (moved from risk_figures.py) ===


@dataclass
class HorizonInfo:
    requested: float
    resolved: float
    index: int


def _load_design_matrix(path: Path) -> pd.DataFrame:
    dm = pd.read_csv(path)
    if EVENT_COL in dm.columns:
        dm = dm[dm[EVENT_COL] != -1].reset_index(drop=True)
    return dm


def _load_survival_predictions(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(path, allow_pickle=True)
    time_key = "time" if "time" in data else "time_grid"
    if time_key not in data or "surv" not in data:
        raise ValueError("Expected `surv` and `time`/`time_grid` in survival npz.")

    time_grid = np.asarray(data[time_key], dtype=float).reshape(-1)
    surv = np.asarray(data["surv"], dtype=float)
    if surv.shape[0] == time_grid.shape[0] and surv.shape[0] != surv.shape[1]:
        surv = surv.T
    if surv.shape[1] != time_grid.shape[0]:
        raise ValueError(
            f"surv shape {surv.shape} incompatible with time grid {time_grid.shape}"
        )
    idx_test = None
    for key in ("idx_test", "idx", "indices"):
        if key in data:
            idx_test = np.asarray(data[key], dtype=int)
            break
    return time_grid, surv, idx_test


def _subtype_label(hr_val: float, her2_val: float) -> str | None:
    if pd.isna(hr_val) or pd.isna(her2_val):
        return None
    hr = "HR+" if int(round(hr_val)) == 1 else "HR-"
    her2 = "HER2+" if int(round(her2_val)) == 1 else "HER2-"
    return f"{hr}/{her2}"


def _assign_tertiles(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    valid = values.dropna()
    strata = pd.Series(index=values.index, dtype=object)
    if valid.empty:
        return strata

    if valid.shape[0] < 3:
        ordered = valid.sort_values()
        if ordered.shape[0] == 1:
            strata.loc[ordered.index[0]] = "low"
            return strata
        strata.loc[ordered.index[0]] = "low"
        strata.loc[ordered.index[1]] = "high"
        return strata

    ranks = valid.rank(method="first")
    n = ranks.shape[0]
    cutoff1 = int(np.ceil(n / 3))
    cutoff2 = int(np.ceil(2 * n / 3))
    ordered_idx = ranks.sort_values().index
    strata.loc[ordered_idx[:cutoff1]] = "low"
    strata.loc[ordered_idx[cutoff1:cutoff2]] = "mid"
    strata.loc[ordered_idx[cutoff2:]] = "high"
    return strata


def _bootstrap_patient_metric(
    durations: np.ndarray,
    events: np.ndarray,
    patient_ids: Sequence[str],
    metric: str,
    rmst_tau: float,
    n_boot: int = 400,
    random_state: int = RANDOM_STATE,
) -> tuple[float, tuple[float | None, float | None]]:
    rng = np.random.default_rng(random_state)
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    patient_ids = pd.Series(patient_ids, dtype=str)

    def _stat(d: np.ndarray, e: np.ndarray) -> float:
        kmf = KaplanMeierFitter()
        kmf.fit(durations=d, event_observed=e)
        if metric == "median":
            val = float(kmf.median_survival_time_)
            if np.isfinite(val):
                return val
        return float(restricted_mean_survival_time(kmf, t=rmst_tau))

    point_estimate = _stat(durations, events)
    if n_boot <= 0:
        return point_estimate, (None, None)

    pid_groups = patient_ids.groupby(patient_ids, sort=False).indices
    unique_pids = np.asarray(list(pid_groups.keys()), dtype=object)
    if unique_pids.size == 0:
        return point_estimate, (None, None)
    pid_indices = [np.asarray(pid_groups[pid], dtype=int) for pid in unique_pids]

    samples: list[float] = []
    for _ in range(n_boot):
        sampled_idx = rng.integers(0, unique_pids.size, size=unique_pids.size)
        boot_indices = np.concatenate([pid_indices[i] for i in sampled_idx])
        if boot_indices.size == 0:
            continue
        samples.append(_stat(durations[boot_indices], events[boot_indices]))

    if not samples:
        return point_estimate, (None, None)
    low, high = np.percentile(samples, [2.5, 97.5])
    return point_estimate, (float(low), float(high))


def _normalize_rmst_taus(
    rmst_tau: float | None,
    rmst_taus: Sequence[float] | None,
) -> tuple[float, list[float]]:
    primary_tau = 365.0 if rmst_tau is None else float(rmst_tau)
    tau_list = [180.0, 365.0, 730.0] if rmst_taus is None else list(rmst_taus)
    if primary_tau not in tau_list:
        tau_list = [primary_tau] + tau_list

    cleaned: list[float] = []
    for tau in tau_list:
        try:
            tau_val = float(tau)
        except (TypeError, ValueError):
            continue
        if np.isfinite(tau_val) and tau_val > 0:
            if tau_val not in cleaned:
                cleaned.append(tau_val)
    if not cleaned:
        raise ValueError("rmst_taus must contain at least one positive value.")
    return primary_tau, cleaned


def _collect_tertile_stratified_survival_data(
    time_grid: Sequence[float],
    surv_np: np.ndarray,
    design_matrix: pd.DataFrame,
    strata_indices: Mapping[str, Sequence[int]],
    risk_scores: Sequence[float] | pd.Series,
    horizon_day: float,
    risk_table_times: Sequence[float],
    n_boot: int,
    random_state: int,
    rmst_tau: float | None,
    rmst_taus: Sequence[float] | None = None,
    pred_ci_method: str = "bootstrap",
) -> tuple[list[dict[str, Any]], list[dict[str, object]], Dict[str, Any]]:
    time_grid = np.asarray(time_grid, dtype=float)
    surv_np = np.asarray(surv_np, dtype=float)
    if surv_np.ndim != 2 or time_grid.ndim != 1:
        raise ValueError("time_grid must match surv_np columns.")
    if surv_np.shape[1] != time_grid.shape[0]:
        raise ValueError("surv_np columns must match time_grid length.")
    horizon_day = float(horizon_day)
    matches = np.isclose(time_grid, horizon_day, atol=1e-6)
    if not np.any(matches):
        raise ValueError(
            "horizon_day must exist exactly in time_grid for pred horizon."
        )
    horizon_idx = int(np.flatnonzero(matches)[0])
    primary_rmst_tau, rmst_taus = _normalize_rmst_taus(rmst_tau, rmst_taus)

    if isinstance(risk_scores, pd.Series):
        risk_series = risk_scores.astype(float).reset_index(drop=True)
    else:
        risk_series = pd.Series(np.asarray(risk_scores, dtype=float))
    if risk_series.shape[0] != surv_np.shape[0]:
        raise ValueError("risk_scores length must match surv_np rows.")

    summary_rows: list[dict[str, object]] = []
    strata_indices_clean: Dict[str, list[int]] = {}
    lr_durations: list[np.ndarray] = []
    lr_events: list[np.ndarray] = []
    lr_groups: list[np.ndarray] = []
    strata_data: list[dict[str, Any]] = []
    rmst_rows: list[dict[str, object]] = []

    pred_ci_method = pred_ci_method.lower().strip()
    if pred_ci_method not in {"quantile", "bootstrap"}:
        raise ValueError("pred_ci_method must be 'quantile' or 'bootstrap'.")

    def _predicted_curve_ci(
        curves: np.ndarray,
        mean_curve: np.ndarray,
        *,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if curves.shape[0] <= 1:
            return mean_curve.copy(), mean_curve.copy()
        if pred_ci_method == "quantile":
            low = np.nanquantile(curves, 0.025, axis=0)
            high = np.nanquantile(curves, 0.975, axis=0)
            return low, high
        if n_boot <= 0:
            return mean_curve.copy(), mean_curve.copy()
        rng = np.random.default_rng(seed)
        n_samples, n_times = curves.shape
        boot_means = np.empty((n_boot, n_times), dtype=float)
        for i in range(n_boot):
            sample_idx = rng.integers(0, n_samples, size=n_samples)
            boot_means[i] = np.nanmean(curves[sample_idx], axis=0)
        low = np.nanquantile(boot_means, 0.025, axis=0)
        high = np.nanquantile(boot_means, 0.975, axis=0)
        return low, high

    base_seed = int(random_state) if random_state is not None else 0
    for stratum_idx, stratum in enumerate(STRATA_LABELS):
        member_idx = np.asarray(strata_indices.get(stratum, []), dtype=int)
        if member_idx.size == 0:
            continue
        mask = (member_idx >= 0) & (member_idx < surv_np.shape[0])
        if not np.any(mask):
            continue
        member_idx = np.unique(member_idx[mask])
        strata_indices_clean[stratum] = member_idx.tolist()

        curves = surv_np[member_idx]
        pred_mean = np.nanmean(curves, axis=0)
        pred_ci_low, pred_ci_high = _predicted_curve_ci(
            curves,
            pred_mean,
            seed=base_seed + 10007 * (stratum_idx + 1),
        )
        pred_ci_low = np.clip(pred_ci_low, 0.0, 1.0)
        pred_ci_high = np.clip(pred_ci_high, 0.0, 1.0)

        rows = design_matrix.iloc[member_idx]
        durations = pd.to_numeric(rows[TIME_COL], errors="coerce").to_numpy(dtype=float)
        events = pd.to_numeric(rows[EVENT_COL], errors="coerce").to_numpy(dtype=float)
        patient_ids = rows[PATIENT_ID_COL].astype(str).to_numpy()
        valid = np.isfinite(durations) & np.isfinite(events)
        durations = durations[valid]
        events = events[valid].astype(int)
        patient_ids = patient_ids[valid]

        rmst_val = float("nan")
        ci: tuple[float | None, float | None] = (None, None)
        km_index = None
        km_values = None
        rmst_km_by_tau = {tau: float("nan") for tau in rmst_taus}
        if durations.size > 0:
            km_index, km_values, kmf = km_curve(
                durations=durations,
                events=events,
                label=f"{stratum} observed KM",
            )
            for tau in rmst_taus:
                rmst_km_by_tau[tau] = float(restricted_mean_survival_time(kmf, t=tau))
            rmst_val, ci = _bootstrap_patient_metric(
                durations=durations,
                events=events,
                patient_ids=patient_ids,
                metric="rmst",
                rmst_tau=primary_rmst_tau,
                n_boot=n_boot,
                random_state=random_state,
            )
            lr_durations.append(durations)
            lr_events.append(events)
            lr_groups.append(np.array([stratum] * durations.size))

        rmst_pred_by_tau = {
            tau: rmst_from_curve(time_grid, pred_mean, tau) for tau in rmst_taus
        }
        for tau in rmst_taus:
            rmst_km = rmst_km_by_tau.get(tau, float("nan"))
            rmst_pred = rmst_pred_by_tau.get(tau, float("nan"))
            rmst_rows.append(
                {
                    "stratum": stratum,
                    "tau": int(round(tau)),
                    "rmst_km": rmst_km,
                    "rmst_pred_mean": rmst_pred,
                    "rmst_diff": rmst_pred - rmst_km,
                }
            )

        risk_subset = risk_series.reindex(member_idx)
        risk_subset = risk_subset.dropna()
        risk_mean = float(risk_subset.mean()) if not risk_subset.empty else float("nan")
        risk_median = (
            float(risk_subset.median()) if not risk_subset.empty else float("nan")
        )
        summary_entry = {
            "stratum": stratum,
            "n": int(member_idx.size),
            "risk_mean": risk_mean,
            "risk_median": risk_median,
            "pred_surv_at_horizon": float(np.nanmean(surv_np[member_idx, horizon_idx])),
            "rmst": float(rmst_val) if np.isfinite(rmst_val) else np.nan,
            "rmst_ci_low": ci[0],
            "rmst_ci_high": ci[1],
            "rmst_pred_mean": rmst_pred_by_tau.get(primary_rmst_tau, float("nan")),
        }
        summary_rows.append(summary_entry)
        strata_data.append(
            {
                "stratum": stratum,
                "color": STRATA_COLORS[stratum],
                "pred_mean": pred_mean,
                "pred_ci_low": pred_ci_low,
                "pred_ci_high": pred_ci_high,
                "km_index": km_index,
                "km_values": km_values,
                "rmst": summary_entry["rmst"],
                "rmst_pred_mean": summary_entry["rmst_pred_mean"],
                "rmst_ci": (
                    summary_entry["rmst_ci_low"],
                    summary_entry["rmst_ci_high"],
                ),
            }
        )

    if not summary_rows:
        raise ValueError("No strata with members were provided.")

    logrank_p: float | None = None
    if len(lr_groups) >= 2:
        try:
            durations_cat = np.concatenate(lr_durations)
            events_cat = np.concatenate(lr_events)
            groups_cat = np.concatenate(lr_groups)
            if (
                durations_cat.size == events_cat.size
                and np.unique(groups_cat).size >= 2
            ):
                lr_result = multivariate_logrank_test(
                    event_durations=durations_cat,
                    groups=groups_cat,
                    event_observed=events_cat.astype(int),
                )
                logrank_p = float(lr_result.p_value)
        except Exception as exc:
            print(f"Warning: log-rank test failed ({exc}).")
            logrank_p = None

    risk_table_times = [float(t) for t in risk_table_times]
    at_risk_rows: list[dict[str, object]] = []
    for stratum, idxs in strata_indices_clean.items():
        if not idxs:
            continue
        rows = design_matrix.iloc[idxs]
        durations = pd.to_numeric(rows[TIME_COL], errors="coerce").to_numpy(dtype=float)
        durations = durations[np.isfinite(durations)]
        row = {"stratum": stratum}
        for tp in risk_table_times:
            row[f"{int(tp)}d_at_risk"] = int(np.sum(durations >= tp))
        at_risk_rows.append(row)
    if at_risk_rows:
        at_risk_df = (
            pd.DataFrame(at_risk_rows)
            .set_index("stratum")
            .reindex(STRATA_LABELS)
            .reset_index()
        )
    else:
        at_risk_df = pd.DataFrame(columns=["stratum"])

    if rmst_rows:
        rmst_compare_df = pd.DataFrame(rmst_rows)
        rmst_compare_df["stratum"] = pd.Categorical(
            rmst_compare_df["stratum"], categories=STRATA_LABELS, ordered=True
        )
        rmst_compare_df = rmst_compare_df.sort_values(["stratum", "tau"]).reset_index(
            drop=True
        )
    else:
        rmst_compare_df = pd.DataFrame(
            columns=["stratum", "tau", "rmst_km", "rmst_pred_mean", "rmst_diff"]
        )

    rmst_taus_label = ", ".join(f"{int(t)}" for t in rmst_taus)
    explanation = (
        f"Tertiles are defined using predicted event probability (1 - survival) at {horizon_day} days. "
        "Dashed lines show the mean predicted survival within each tertile; solid lines show the observed "
        "Kaplan–Meier curves for the same patients. Restricted mean survival time (RMST) is the area under "
        f"the observed KM curve up to {int(primary_rmst_tau)} days for the reported CIs, while RMST is also "
        f"reported for the mean predicted curve at taus {{{rmst_taus_label}}} days. The reported CIs are the "
        f"2.5th/97.5th percentiles from {n_boot} patient-level bootstrap resamples (resampling on patient_id "
        "preserves within-patient lines). "
        "The log-rank test is a global comparison of the observed survival distributions across strata."
    )
    if logrank_p is not None:
        explanation += f" log-rank p={logrank_p:.3g}."

    details: Dict[str, Any] = {
        "strata_indices": strata_indices_clean,
        "horizon_day": horizon_day,
        "rmst_tau": primary_rmst_tau,
        "rmst_taus": rmst_taus,
        "logrank_p": logrank_p,
        "pred_ci_method": pred_ci_method,
        "explanation": explanation,
        "at_risk_table": at_risk_df,
        "rmst_comparison": rmst_compare_df,
    }
    return strata_data, summary_rows, details


def plot_tertile_stratified_survival(
    time_grid: Sequence[float],
    surv_np: np.ndarray,
    design_matrix: pd.DataFrame,
    output_path: Union[str, Path],
    strata_indices: Mapping[str, Sequence[int]],
    risk_scores: Sequence[float] | pd.Series,
    horizon_day: float,
    rmst_tau: float,
    rmst_taus: Sequence[float],
    risk_table_times: Sequence[float],
    n_boot: int,
    random_state: int,
) -> tuple[Path, pd.DataFrame, Dict[str, Any]]:
    """Plot predicted vs observed survival for explicit strata with RMST/CI/log-rank."""
    plt.rcParams.update(STYLE)
    strata_data, summary_rows, details = _collect_tertile_stratified_survival_data(
        time_grid=time_grid,
        surv_np=surv_np,
        design_matrix=design_matrix,
        strata_indices=strata_indices,
        risk_scores=risk_scores,
        horizon_day=horizon_day,
        rmst_tau=rmst_tau,
        rmst_taus=rmst_taus,
        risk_table_times=risk_table_times,
        n_boot=n_boot,
        random_state=random_state,
    )

    fig, (ax, ax_table) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8.6, 7.4),
        gridspec_kw={"height_ratios": [3.2, 1.3]},
    )
    for row in strata_data:
        ax.step(
            time_grid,
            row["pred_mean"],
            where="post",
            color=row["color"],
            lw=2.2,
            ls="--",
            alpha=0.9,
            label=f"{row['stratum']} predicted (mean)",
        )
        if row["km_index"] is not None and row["km_values"] is not None:
            ax.step(
                row["km_index"],
                row["km_values"],
                where="post",
                color=row["color"],
                lw=2.4,
                ls="-",
                alpha=0.95,
                label=f"{row['stratum']} observed KM",
            )

    ax.set_xlabel("Days since index")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Tertiles by predicted risk @ {horizon_day}d", pad=8)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, ncol=2)

    summary_text: list[str] = []
    for row in summary_rows:
        ci_low, ci_high = row["rmst_ci_low"], row["rmst_ci_high"]
        ci_str = (
            f" [{ci_low:.0f}, {ci_high:.0f}]"
            if ci_low is not None and ci_high is not None
            else ""
        )
        if np.isfinite(row["rmst"]):
            summary_text.append(
                f"{row['stratum']}: RMST@{int(details['rmst_tau'])}={row['rmst']:.0f}{ci_str}"
            )
    # if summary_text:
    #     ax.text(
    #         0.02,
    #         0.02,
    #         "\n".join(summary_text),
    #         ha="left",
    #         va="bottom",
    #         fontsize=9,
    #         transform=ax.transAxes,
    #     )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.set_index("stratum").reindex(STRATA_LABELS).reset_index()
    summary_df["n"] = summary_df["n"].fillna(0).astype(int)
    summary_df["mae_pred_vs_km"] = summary_df["stratum"].map(mae_by_label)

    rmst_compare = details.get("rmst_comparison")
    if isinstance(rmst_compare, pd.DataFrame) and not rmst_compare.empty:
        print("\nRMST comparison (KM vs mean predicted survival):")
        display_df = rmst_compare.copy()
        for col in ("rmst_km", "rmst_pred_mean", "rmst_diff"):
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.0f}" if np.isfinite(x) else "nan"
            )
        print(display_df.to_string(index=False))
    return output_path, summary_df, details


def plot_tertile_stratified_survival_and_km(
    time_grid: Sequence[float],
    surv_np: np.ndarray,
    design_matrix: pd.DataFrame,
    output_path: Union[str, Path],
    strata_indices: Mapping[str, Sequence[int]] | None,
    risk_scores: Sequence[float] | pd.Series,
    horizon_day: float,
    rmst_tau: float,
    rmst_taus: Sequence[float],
    risk_table_times: Sequence[float],
    n_boot: int,
    random_state: int,
    subset_indices: Sequence[int] | Sequence[bool] | None = None,
    cohort_label: str = "population",
    show_table: bool = True,
    pred_ci_method: str = "bootstrap",
) -> tuple[Path, pd.DataFrame, Dict[str, Any]]:
    """Plot stratified survival curves with population KM/mean prediction and summary tables."""
    plt.rcParams.update(STYLE)
    if subset_indices is not None:
        subset_labels = None
        subset_arr = np.asarray(subset_indices)
        if subset_arr.dtype == bool:
            if subset_arr.shape[0] != len(design_matrix):
                raise ValueError("subset_indices mask must match design_matrix length.")
            subset_idx = np.flatnonzero(subset_arr)
        else:
            subset_list = list(dict.fromkeys(subset_indices))
            if not subset_list:
                raise ValueError("subset_indices cannot be empty.")
            indexer = design_matrix.index.get_indexer(subset_list)
            if np.all(indexer >= 0):
                subset_idx = indexer
                subset_labels = subset_list
            else:
                try:
                    subset_idx = np.asarray(subset_list, dtype=int)
                except Exception as exc:
                    raise ValueError(
                        "subset_indices must be a boolean mask, index labels, or integer positions."
                    ) from exc
                subset_idx = subset_idx[
                    (subset_idx >= 0) & (subset_idx < len(design_matrix))
                ]
        if subset_idx.size == 0:
            raise ValueError("subset_indices produced no valid rows.")
        subset_idx = np.asarray(
            list(dict.fromkeys(int(idx) for idx in subset_idx)), dtype=int
        )
        if subset_labels is not None:
            design_matrix = design_matrix.loc[subset_labels]
            if isinstance(surv_np, pd.DataFrame):
                surv_np = surv_np.loc[subset_labels]
            else:
                surv_np = np.asarray(surv_np)[subset_idx]
            if isinstance(risk_scores, pd.Series):
                risk_scores = risk_scores.loc[subset_labels]
            else:
                risk_scores = np.asarray(risk_scores)[subset_idx]
        else:
            design_matrix = design_matrix.iloc[subset_idx]
            if isinstance(surv_np, pd.DataFrame):
                surv_np = surv_np.iloc[subset_idx]
            else:
                surv_np = np.asarray(surv_np)[subset_idx]
            if isinstance(risk_scores, pd.Series):
                risk_scores = risk_scores.iloc[subset_idx]
            else:
                risk_scores = np.asarray(risk_scores)[subset_idx]

        if strata_indices is None:
            strata = _assign_tertiles(pd.Series(np.asarray(risk_scores, dtype=float)))
            strata_indices = {
                label: strata[strata == label].index.to_list()
                for label in STRATA_LABELS
                if (strata == label).any()
            }
        else:
            strata_indices = _remap_strata_indices(strata_indices, subset_idx)
    elif strata_indices is None:
        strata = _assign_tertiles(pd.Series(np.asarray(risk_scores, dtype=float)))
        strata_indices = {
            label: strata[strata == label].index.to_list()
            for label in STRATA_LABELS
            if (strata == label).any()
        }

    strata_data, summary_rows, details = _collect_tertile_stratified_survival_data(
        time_grid=time_grid,
        surv_np=surv_np,
        design_matrix=design_matrix,
        strata_indices=strata_indices,
        risk_scores=risk_scores,
        horizon_day=horizon_day,
        rmst_tau=rmst_tau,
        rmst_taus=rmst_taus,
        risk_table_times=risk_table_times,
        n_boot=n_boot,
        random_state=random_state,
        pred_ci_method=pred_ci_method,
    )
    logrank_p = details.get("logrank_p")
    if logrank_p is not None:
        print(f"Log-rank p-value ({cohort_label}): {logrank_p:.3g}")

    time_grid = np.asarray(time_grid, dtype=float)
    surv_np = np.asarray(surv_np, dtype=float)
    mean_curve = np.nanmean(surv_np, axis=0)

    durations_all = pd.to_numeric(design_matrix[TIME_COL], errors="coerce").to_numpy(
        dtype=float
    )
    events_all = pd.to_numeric(design_matrix[EVENT_COL], errors="coerce").to_numpy(
        dtype=float
    )
    valid_all = np.isfinite(durations_all) & np.isfinite(events_all)
    durations_all = durations_all[valid_all]
    events_all = events_all[valid_all].astype(int)

    kmf_all = None
    if durations_all.size > 0:
        km_all_index, km_all_values, kmf_all = km_curve(
            durations=durations_all,
            events=events_all,
            label="Population observed KM",
        )
    else:
        km_all_index = np.array([], dtype=float)
        km_all_values = np.array([], dtype=float)

    if show_table:
        fig, (ax, ax_table) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8.6, 7.4),
            gridspec_kw={"height_ratios": [3.2, 1.3]},
        )
    else:
        fig, ax = plt.subplots(figsize=(8.6, 5.8))
        ax_table = None
    for row in strata_data:
        if row.get("pred_ci_low") is not None and row.get("pred_ci_high") is not None:
            ax.fill_between(
                time_grid,
                row["pred_ci_low"],
                row["pred_ci_high"],
                step="post",
                color=row["color"],
                alpha=0.12,
                linewidth=0.0,
            )
        ax.step(
            time_grid,
            row["pred_mean"],
            where="post",
            color=row["color"],
            lw=1.8,
            ls="--",
            alpha=0.8,
            label="_nolegend_",
        )
        if row["km_index"] is not None and row["km_values"] is not None:
            ax.step(
                row["km_index"],
                row["km_values"],
                where="post",
                color=row["color"],
                lw=2.2,
                ls="-",
                alpha=0.9,
                label="_nolegend_",
            )

    population_color = "#b91c1c"
    ax.step(
        time_grid,
        mean_curve,
        where="post",
        color=population_color,
        lw=2.4,
        ls="--",
        alpha=0.9,
        label="_nolegend_",
    )
    if km_all_index.size:
        ax.step(
            km_all_index,
            km_all_values,
            where="post",
            color=population_color,
            lw=2.8,
            ls="-",
            alpha=0.95,
            label="_nolegend_",
        )

    mae_by_label: dict[str, float] = {}
    mae_by_label["Population"] = mae_pred_vs_km(
        time_grid, mean_curve, km_all_index, km_all_values
    )
    for row in strata_data:
        if row["km_index"] is None or row["km_values"] is None:
            mae_by_label[row["stratum"]] = float("nan")
        else:
            mae_by_label[row["stratum"]] = mae_pred_vs_km(
                time_grid, row["pred_mean"], row["km_index"], row["km_values"]
            )

    def _format_mae(val: float) -> str:
        return "NA" if not np.isfinite(val) else f"{val:.3f}"

    ax.set_xlabel("Days since index")
    ax.set_ylabel("Survival probability")
    ax.set_xlim(SURVIVAL_X_LIMIT)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Risk-stratified survival curves for {cohort_label} cohort", pad=8)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    stratum_label_map = {
        "low": "Low-risk",
        "mid": "Mid-risk",
        "high": "High-risk",
        "Population": "Population",
    }
    legend_handles = [
        Line2D(
            [],
            [],
            color="#111827",
            lw=2.0,
            ls="--",
            label="Prediction (dashed)",
        ),
        Line2D(
            [],
            [],
            color="#111827",
            lw=2.0,
            ls="-",
            label="Observed KM (solid)",
        ),
        Line2D(
            [],
            [],
            color=population_color,
            lw=2.6,
            label=f"Population (MAE={_format_mae(mae_by_label.get('Population', float('nan')))})",
        ),
        Line2D(
            [],
            [],
            color=STRATA_COLORS["low"],
            lw=2.4,
            label=(
                f"{stratum_label_map['low']} "
                f"(MAE={_format_mae(mae_by_label.get('low', float('nan')))})"
            ),
        ),
        Line2D(
            [],
            [],
            color=STRATA_COLORS["mid"],
            lw=2.4,
            label=(
                f"{stratum_label_map['mid']} "
                f"(MAE={_format_mae(mae_by_label.get('mid', float('nan')))})"
            ),
        ),
        Line2D(
            [],
            [],
            color=STRATA_COLORS["high"],
            lw=2.4,
            label=(
                f"{stratum_label_map['high']} "
                f"(MAE={_format_mae(mae_by_label.get('high', float('nan')))})"
            ),
        ),
    ]
    ax.legend(handles=legend_handles, frameon=False, ncol=2, loc="upper right")

    horizon_day_val = float(details.get("horizon_day", horizon_day))
    horizon_idx = int(np.argmin(np.abs(time_grid - horizon_day_val)))
    resolved_horizon = float(time_grid[horizon_idx])
    stats_lines = [f"Predicted S({int(resolved_horizon)}d)"]
    for stratum in STRATA_LABELS:
        idxs = details.get("strata_indices", {}).get(stratum, [])
        if not idxs:
            continue
        vals = surv_np[np.asarray(idxs, dtype=int), horizon_idx]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean_val = float(np.mean(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        label = stratum_label_map.get(stratum, str(stratum))
        stats_lines.append(f"{label}: {mean_val:.2f} [{min_val:.2f}-{max_val:.2f}]")
    if len(stats_lines) > 1:
        ax.text(
            0.02,
            0.02,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="#111827",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=3.0),
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.set_index("stratum").reindex(STRATA_LABELS).reset_index()
    summary_df["n"] = summary_df["n"].fillna(0).astype(int)

    rmst_compare = details.get("rmst_comparison")
    if isinstance(rmst_compare, pd.DataFrame) and not rmst_compare.empty:
        print("\nRMST comparison (KM vs mean predicted survival):")
        display_df = rmst_compare.copy()
        for col in ("rmst_km", "rmst_pred_mean", "rmst_diff"):
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.0f}" if np.isfinite(x) else "nan"
            )
        print(display_df.to_string(index=False))

    combined_table = (
        summary_df[["stratum", "n"]]
        .merge(details["at_risk_table"], on="stratum", how="left")
        .rename(columns={"n": "count"})
    )
    rmst_tau_val = int(round(float(details.get("rmst_tau", rmst_tau))))
    if isinstance(rmst_compare, pd.DataFrame) and not rmst_compare.empty:
        rmst_tau_df = rmst_compare.loc[
            rmst_compare["tau"].astype(int) == rmst_tau_val,
            ["stratum", "rmst_km", "rmst_pred_mean", "rmst_diff"],
        ].rename(
            columns={
                "rmst_km": "RMST_KM",
                "rmst_pred_mean": "RMST prediction",
                "rmst_diff": "RMST delta",
            }
        )
        combined_table = combined_table.merge(rmst_tau_df, on="stratum", how="left")

    pop_row: dict[str, object] = {
        "stratum": "Population",
        "count": int(surv_np.shape[0]),
    }
    for tp in risk_table_times:
        pop_row[f"{int(tp)}d_at_risk"] = int(np.sum(durations_all >= float(tp)))
    if durations_all.size and kmf_all is not None:
        pop_row["RMST_KM"] = float(
            restricted_mean_survival_time(kmf_all, t=rmst_tau_val)
        )
    else:
        pop_row["RMST_KM"] = float("nan")
    pop_row["RMST prediction"] = rmst_from_curve(time_grid, mean_curve, rmst_tau_val)
    pop_row["RMST delta"] = float(pop_row["RMST prediction"]) - float(
        pop_row["RMST_KM"]
    )

    combined_table = pd.concat(
        [pd.DataFrame([pop_row]), combined_table], ignore_index=True
    )
    combined_table["stratum"] = combined_table["stratum"].map(
        lambda s: stratum_label_map.get(s, s)
    )
    combined_table = combined_table.drop(columns=["90d_at_risk"], errors="ignore")
    at_risk_cols = [
        c
        for c in combined_table.columns
        if re.fullmatch(r"\d+d_at_risk", str(c)) is not None
        and not str(c).startswith("90d")
    ]
    at_risk_cols = sorted(at_risk_cols, key=lambda c: int(str(c).split("d_at_risk")[0]))
    ordered_cols = (
        ["stratum", "count"]
        + at_risk_cols
        + [
            "RMST_KM",
            "RMST delta",
        ]
    )
    combined_table = combined_table[
        [c for c in ordered_cols if c in combined_table.columns]
    ]
    numeric_cols = [c for c in combined_table.columns if c != "stratum"]
    for col in numeric_cols:
        combined_table[col] = combined_table[col].round(0).astype(pd.Int64Dtype())

    details["risk_summary_table"] = combined_table
    details["population_metrics"] = {
        "n": int(surv_np.shape[0]),
        "rmst_km": float(pop_row["RMST_KM"]),
        "rmst_pred_mean": float(pop_row["RMST prediction"]),
        "mae_pred_vs_km": float(mae_by_label.get("Population", float("nan"))),
    }
    details["mae_by_stratum"] = {
        k: float(v) for k, v in mae_by_label.items() if k in STRATA_LABELS
    }

    display_cols = combined_table.columns.tolist()
    header_labels = {
        "stratum": "Stratum",
        "count": "Count",
        "RMST_KM": "KM RMST",
        "RMST delta": "delta KMST",
    }
    col_labels = [
        header_labels.get(col, col.replace("_at_risk", "").replace("d", " d"))
        for col in display_cols
    ]
    table_df = combined_table.copy()
    for col in display_cols:
        if col == "stratum":
            continue
        table_df[col] = table_df[col].apply(
            lambda x: "" if pd.isna(x) else f"{int(x):,}"
        )
    cell_text = table_df[display_cols].values.tolist()

    if show_table and ax_table is not None:
        ax_table.axis("off")
        table = ax_table.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="right",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12.5)
        table.scale(1.1, 1.6)
        last_row = max(row for row, _ in table.get_celld().keys())
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#111827")
            cell.set_linewidth(0.8)
            cell.visible_edges = ""
            if row_idx == 0:
                cell.set_facecolor("#f3f4f6")
                cell.visible_edges = "TB"
                if col_idx == 0:
                    cell.set_text_props(weight="bold", color="#111827", ha="left")
                else:
                    cell.set_text_props(weight="bold", color="#111827", ha="right")
            else:
                cell.set_facecolor("#fafafa" if row_idx % 2 == 1 else "white")
                if row_idx == last_row:
                    cell.visible_edges = "B"
                if col_idx == 0:
                    cell.set_text_props(ha="left")
                else:
                    cell.set_text_props(ha="right")
            cell.PAD = 0.28

        fig.subplots_adjust(hspace=0.18)
    out_path = _ensure_png_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400)
    plt.close(fig)

    print(
        "\nRisk-set counts (at risk) and RMST, used for `sn-article.tex`: Table `table:risk-group-diffs`."
    )
    print(combined_table.to_string(index=False))

    return out_path, summary_df, details


def _remap_strata_indices(
    strata_indices: Mapping[str, Sequence[int]],
    subset_idx: Sequence[int],
) -> Dict[str, list[int]]:
    subset_idx_arr = np.asarray(subset_idx, dtype=int).ravel()
    ordered_idx = list(dict.fromkeys(int(idx) for idx in subset_idx_arr))
    mapping = {idx: new for new, idx in enumerate(ordered_idx)}
    remapped: Dict[str, list[int]] = {}
    for label, indices in strata_indices.items():
        dest = []
        for idx in indices:
            mapped = mapping.get(int(idx))
            if mapped is not None:
                dest.append(mapped)
        if dest:
            remapped[label] = dest
    return remapped


def plot_tertile_stratified_survival_by_subtype(
    time_grid: Sequence[float],
    surv_np: np.ndarray,
    design_matrix: pd.DataFrame,
    strata_indices: Mapping[str, Sequence[int]],
    risk_scores: Sequence[float],
    group_indices: Mapping[str, Sequence[int]],
    horizon_day: float,
    rmst_tau: float,
    rmst_taus: Sequence[float],
    risk_table_times: Sequence[float],
    n_boot: int,
    random_state: int,
    output_path: Union[str, Path],
    group_order: Sequence[str] | None = None,
    figsize: tuple[float, float] = (11.0, 9.0),
    dpi: int = 400,
) -> Path:
    plt.rcParams.update(STYLE)
    time_grid = np.asarray(time_grid, dtype=float)
    surv_np = np.asarray(surv_np, dtype=float)
    risk_arr = np.asarray(risk_scores, dtype=float)
    if risk_arr.ndim != 1:
        raise ValueError("risk_scores must be a 1D sequence.")
    if risk_arr.shape[0] != surv_np.shape[0]:
        raise ValueError("risk_scores must align with surv_np rows.")

    ordered_groups = list(group_order or group_indices.keys())
    if len(ordered_groups) > 4:
        ordered_groups = ordered_groups[:4]
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    axes_flat = axes.ravel()

    for idx, label in enumerate(ordered_groups):
        ax = axes_flat[idx]
        subset_idx = np.asarray(group_indices.get(label, []), dtype=int)
        subset_idx = subset_idx[(subset_idx >= 0) & (subset_idx < surv_np.shape[0])]
        subset_idx = np.unique(subset_idx)
        if subset_idx.size == 0:
            ax.set_facecolor("#ffffff")
            ax.text(
                0.5,
                0.5,
                "No patients",
                ha="center",
                va="center",
                fontsize=11,
                color="#444444",
            )
            ax.set_title(label, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        subset_surv = surv_np[subset_idx]
        subset_design = design_matrix.iloc[subset_idx]
        subset_risk = risk_arr[subset_idx]
        subset_strata = _remap_strata_indices(strata_indices, subset_idx)
        strata_data, summary_rows, _ = _collect_tertile_stratified_survival_data(
            time_grid=time_grid,
            surv_np=subset_surv,
            design_matrix=subset_design,
            strata_indices=subset_strata,
            risk_scores=subset_risk,
            horizon_day=horizon_day,
            rmst_tau=rmst_tau,
            rmst_taus=rmst_taus,
            risk_table_times=risk_table_times,
            n_boot=n_boot,
            random_state=random_state,
        )
        for row in strata_data:
            ax.step(
                time_grid,
                row["pred_mean"],
                where="post",
                color=row["color"],
                lw=2.0,
                ls="--",
                alpha=0.85,
                label=f"{row['stratum']} predicted",
            )
            if row["km_index"] is not None and row["km_values"] is not None:
                ax.step(
                    row["km_index"],
                    row["km_values"],
                    where="post",
                    color=row["color"],
                    lw=2.2,
                    ls="-",
                    alpha=0.9,
                    label=f"{row['stratum']} observed",
                )
        ax.set_xlabel("Days since index")
        ax.set_ylabel("Survival probability")
        ax.set_ylim(0, 1.02)
        ax.set_title(label, fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, labels, frameon=False, fontsize=8, ncol=2, loc="upper right"
            )
        summary_text: list[str] = []
        rmst_display = rmst_tau if rmst_tau is not None else 365
        for row in summary_rows:
            ci_low, ci_high = row["rmst_ci_low"], row["rmst_ci_high"]
            ci_str = (
                f" [{ci_low:.0f}, {ci_high:.0f}]"
                if ci_low is not None and ci_high is not None
                else ""
            )
            if np.isfinite(row["rmst"]):
                summary_text.append(
                    f"{row['stratum']}: RMST@{int(rmst_display)}={row['rmst']:.0f}{ci_str}"
                )
        # if summary_text:
        #     ax.text(
        #         0.02,
        #         0.02,
        #         "\n".join(summary_text),
        #         ha="left",
        #         va="bottom",
        #         fontsize=8,
        #         transform=ax.transAxes,
        #     )

    for idx in range(len(ordered_groups), axes_flat.size):
        axes_flat[idx].axis("off")

    fig.tight_layout()
    target = _ensure_png_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target


def plot_population_km_vs_mean(
    time_grid: Sequence[float],
    surv_np: np.ndarray,
    indices: Sequence[int],
    durations: Sequence[float],
    events: Sequence[int],
    output_path: Union[str, Path],
    title: str,
    figsize: tuple[float, float] = (8.0, 5.0),
    dpi: int = 400,
) -> tuple[Path, Dict[str, float]]:
    """Plot the population KM curve alongside the mean predicted survival and return MAE."""
    plt.rcParams.update(STYLE)
    time_grid = np.asarray(time_grid, dtype=float)
    surv_np = np.asarray(surv_np, dtype=float)
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        raise ValueError("indices cannot be empty for plotting.")
    if surv_np.shape[1] != time_grid.shape[0]:
        raise ValueError("surv_np columns must match time_grid length.")

    curves = surv_np[idx]
    if curves.ndim != 2:
        raise ValueError("surv_np[indices] must yield 2D array of curves.")
    mean_curve = np.nanmean(curves, axis=0)

    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    if durations.shape[0] != surv_np.shape[0] or events.shape[0] != surv_np.shape[0]:
        raise ValueError("durations/events must align with surv_np rows.")
    durations = durations[idx]
    events = events[idx]

    kmf = KaplanMeierFitter()
    kmf.fit(durations=durations, event_observed=events, label="Observed KM", alpha=0.05)
    observed_series = kmf.survival_function_at_times(time_grid)
    observed_curve = np.asarray(
        [float(observed_series.get(t, np.nan)) for t in time_grid],
        dtype=float,
    )
    ci = kmf.confidence_interval_.sort_index()
    ci_times = ci.index.to_numpy(dtype=float)
    ci_lower = ci.iloc[:, 0].to_numpy(dtype=float)
    ci_upper = ci.iloc[:, 1].to_numpy(dtype=float)

    n_total = int(durations.shape[0])
    at_risk_stats: dict[str, float] = {}
    for t in (180.0, 365.0, 730.0):
        n_at_risk = int(np.sum(durations >= t))
        pct_at_risk = (100.0 * n_at_risk / n_total) if n_total else float("nan")
        at_risk_stats[f"n_at_risk_{int(t)}"] = float(n_at_risk)
        at_risk_stats[f"pct_at_risk_{int(t)}"] = float(pct_at_risk)
    print(
        f"[{title}] N={n_total} | "
        f"At risk @180: {int(at_risk_stats['n_at_risk_180'])} ({at_risk_stats['pct_at_risk_180']:.1f}%) | "
        f"@365: {int(at_risk_stats['n_at_risk_365'])} ({at_risk_stats['pct_at_risk_365']:.1f}%) | "
        f"@730: {int(at_risk_stats['n_at_risk_730'])} ({at_risk_stats['pct_at_risk_730']:.1f}%)"
    )

    mask = np.isfinite(observed_curve) & np.isfinite(mean_curve)
    mae = (
        float(np.nanmean(np.abs(mean_curve[mask] - observed_curve[mask])))
        if np.any(mask)
        else float("nan")
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    mean_line = ax.step(
        time_grid,
        mean_curve,
        where="post",
        label="Mean predicted",
        color=PROJECT_COLOR_CYCLE[0],
        lw=3.0,
    )[0]
    ci_patch = ax.fill_between(
        ci_times,
        ci_lower,
        ci_upper,
        step="post",
        color=PROJECT_COLOR_CYCLE[1],
        alpha=0.2,
        label="KM 95% CI",
    )
    km_line = ax.step(
        observed_series.index,
        observed_series,
        where="post",
        label="Observed KM",
        color=PROJECT_COLOR_CYCLE[1],
        lw=2.6,
        ls="--",
    )[0]
    ax.set_xlabel("Days since index")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    risk_labels = [
        f"At risk @180d: {int(at_risk_stats['n_at_risk_180'])} ({at_risk_stats['pct_at_risk_180']:.1f}%)",
        f"At risk @365d: {int(at_risk_stats['n_at_risk_365'])} ({at_risk_stats['pct_at_risk_365']:.1f}%)",
        f"At risk @730d: {int(at_risk_stats['n_at_risk_730'])} ({at_risk_stats['pct_at_risk_730']:.1f}%)",
    ]
    risk_handles = [Line2D([], [], color="none", label=label) for label in risk_labels]
    legend_handles = [mean_line, km_line, ci_patch, *risk_handles]
    if np.isfinite(mae):
        legend_handles.append(
            Line2D([], [], color="none", label=f"MAE vs KM: {mae:.3f}")
        )
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right",
    )
    fig.tight_layout()

    target = _ensure_png_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target, {"mae": mae, **at_risk_stats}


def plot_predicted_summary_curves(
    time_grid: Sequence[float],
    surv_np: np.ndarray,
    indices: Sequence[int],
    risk_scores: Sequence[float],
    risk_horizon: float,
    include_extremes: bool,
    output_path: Union[str, Path],
    title: str,
    figsize: tuple[float, float],
    dpi: int,
) -> Path:
    """Plot mean predicted survival with optional lowest/highest risk curves."""
    plt.rcParams.update(STYLE)
    time_grid = np.asarray(time_grid, dtype=float)
    surv_np = np.asarray(surv_np, dtype=float)
    idx = np.asarray(indices, dtype=int)
    curves = surv_np[idx]
    if curves.shape[1] != time_grid.shape[0]:
        raise ValueError("curves must align with time_grid.")
    risk_scores = np.asarray(risk_scores, dtype=float)
    if risk_scores.shape[0] != idx.shape[0]:
        raise ValueError("risk_scores length must match indices length.")

    fig, ax = plt.subplots(figsize=figsize)
    mean_curve = np.nanmean(curves, axis=0)
    ax.step(time_grid, mean_curve, where="post", color="#1b3a4b", lw=2.6, label="Mean")

    if include_extremes:
        ordering = np.argsort(risk_scores)
        lowest_idx = ordering[0]
        highest_idx = ordering[-1]
        ax.step(
            time_grid,
            curves[lowest_idx],
            where="post",
            color="#2a7f62",
            lw=2.0,
            ls="--",
            alpha=0.9,
            label=f"Lowest risk (idx={idx[lowest_idx]})",
        )
        ax.step(
            time_grid,
            curves[highest_idx],
            where="post",
            color="#b30f24",
            lw=2.0,
            ls="--",
            alpha=0.9,
            label=f"Highest risk (idx={idx[highest_idx]})",
        )
        ax.axvline(
            risk_horizon, color="#444444", lw=1.0, ls=":", alpha=0.75, label=None
        )

    ax.set_xlabel("Days since index")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = _ensure_png_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_stratum_summary_survival(
    risk_df: pd.DataFrame,
    strat_col: str,
    time_grid: Sequence[float],
    surv_np: np.ndarray,
    output_path: Union[str, Path],
    summary: str = "mean",
    risk_table_times: Sequence[float] = (0.0, 180.0, 365.0, 730.0),
    population_df: Optional[pd.DataFrame] = None,
    population_label: str = "Population",
) -> tuple[Path, Dict[str, float], pd.DataFrame]:
    """Plot summary survival curves (predicted vs observed) for each stratum."""
    reducer = np.nanmedian if str(summary).lower() == "median" else np.nanmean
    label_summary = "median" if reducer is np.nanmedian else "mean"
    time_grid = np.asarray(time_grid, dtype=float)
    surv_np = np.asarray(surv_np, dtype=float)
    if surv_np.shape[1] != time_grid.shape[0]:
        raise ValueError("surv_np columns must match time_grid length.")

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    calibration_mae: Dict[str, float] = {}

    risk_table_rows: list[dict[str, object]] = []
    risk_table_times = [float(tp) for tp in risk_table_times]

    def _at_risk_counts(durations: np.ndarray) -> list[int]:
        return [int(np.sum(durations >= tp)) for tp in risk_table_times]

    for stratum in STRATA_LABELS:
        mask = risk_df[strat_col] == stratum
        if not mask.any():
            continue
        local_idx = risk_df.loc[mask, "local_idx"].to_numpy(dtype=int, copy=True)
        curves = surv_np[local_idx]
        summary_curve = reducer(curves, axis=0)
        ax.step(
            time_grid,
            summary_curve,
            where="post",
            color=STRATA_COLORS[stratum],
            lw=2.0,
            alpha=0.95,
            label=f"{stratum} predicted ({label_summary})",
        )

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=risk_df.loc[mask, TIME_COL],
            event_observed=risk_df.loc[mask, EVENT_COL],
        )
        observed_series = kmf.survival_function_at_times(time_grid)
        observed_curve = np.asarray(observed_series, dtype=float)
        ax.step(
            time_grid,
            observed_curve,
            where="post",
            color=STRATA_COLORS[stratum],
            lw=1.6,
            ls="--",
            alpha=0.9,
            label=f"{stratum} observed",
        )
        calibration_mae[stratum] = float(
            np.nanmean(np.abs(summary_curve - observed_curve))
        )

        # build at-risk counts for specified timepoints
        row = {"stratum": stratum}
        durations = risk_df.loc[mask, TIME_COL].to_numpy(dtype=float, copy=True)
        at_risk_counts = _at_risk_counts(durations)
        for tp, count in zip(risk_table_times, at_risk_counts):
            row[f"{int(tp)}d_at_risk"] = count
        risk_table_rows.append(row)

    ax.set_xlabel("Days since index")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Predicted vs observed {label_summary} survival by stratum")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    risk_table_df = pd.DataFrame(risk_table_rows)
    if population_df is not None:
        pop_durations = population_df[TIME_COL].to_numpy(dtype=float, copy=True)
        population_row = {
            "stratum": population_label,
        }
        for tp, count in zip(risk_table_times, _at_risk_counts(pop_durations)):
            population_row[f"{int(tp)}d_at_risk"] = count
        risk_table_df = pd.concat(
            [risk_table_df, pd.DataFrame([population_row])],
            ignore_index=True,
        )

    risk_table_df = risk_table_df.set_index("stratum")
    # Maintain consistent ordering for strata rows.
    ordered_rows = [s for s in STRATA_LABELS if s in risk_table_df.index]
    trailing = [idx for idx in risk_table_df.index if idx not in ordered_rows]
    risk_table_df = risk_table_df.loc[ordered_rows + trailing]

    return out_path, calibration_mae, risk_table_df


def build_patient_first_lot_strata(
    risk_df: pd.DataFrame,
    risk_col: str,
    patient_id_col: str = PATIENT_ID_COL,
    line_col: str = LINE_COL,
    line_start_col: str = "LINE_START",
    output_col: str = "stratum_plan_b_first_lot",
) -> pd.DataFrame:
    cols = [patient_id_col, "subtype", risk_col]
    if line_col in risk_df.columns:
        cols.append(line_col)
    if line_start_col in risk_df.columns:
        cols.append(line_start_col)
    base = risk_df[cols].dropna(subset=[patient_id_col, "subtype", risk_col]).copy()

    sort_cols: list[str] = [patient_id_col]
    if line_col in base.columns:
        base[line_col] = pd.to_numeric(base[line_col], errors="coerce")
        sort_cols.append(line_col)
    if line_start_col in base.columns:
        base[line_start_col] = pd.to_numeric(base[line_start_col], errors="coerce")
        sort_cols.append(line_start_col)

    base = base.sort_values(sort_cols, na_position="last")
    first = base.drop_duplicates(subset=[patient_id_col], keep="first").copy()
    first[output_col] = _assign_tertiles(first[risk_col])
    return first


def _build_feature_to_group(
    features_dict: Mapping[str, Sequence[str]],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for group, feats in features_dict.items():
        for feat in feats:
            mapping[str(feat)] = str(group)
    return mapping


def grouped_shap_contributions(
    shap_path: Path,
    features_dict_path: Path,
    design_matrix: pd.DataFrame,
    idx_test: np.ndarray,
    normalize: bool = True,
    n_boot: int = 400,
    random_state: int = RANDOM_STATE,
    use_manuscript_groups: bool = True,
) -> tuple[pd.DataFrame, Dict[str, tuple[float | None, float | None]]]:
    data = np.load(shap_path, allow_pickle=True)
    shap_values = np.asarray(data["shap_values"], dtype=float)
    feature_names = [str(x) for x in data["feature_names"]]
    sample_idx = np.asarray(data["sample_idx"], dtype=int)
    if shap_values.shape[0] != sample_idx.shape[0]:
        raise ValueError("SHAP rows must align with sample_idx length.")
    if idx_test is None or idx_test.size == 0:
        raise ValueError("idx_test is required to map SHAP rows to patients.")
    # map SHAP samples to global design-matrix indices
    global_idx = np.asarray(sample_idx, dtype=int)
    if np.max(global_idx) >= design_matrix.shape[0]:
        raise ValueError("sample_idx exceeds available design matrix rows.")
    patient_ids = design_matrix.iloc[global_idx][PATIENT_ID_COL].astype(str).to_numpy()

    if use_manuscript_groups:
        feat_to_group = {name: feature_module(name) for name in feature_names}
    else:
        with open(features_dict_path, "r") as fin:
            features_dict = json.load(fin)
        feat_to_group = _build_feature_to_group(features_dict)

    groups: list[str] = []
    feature_group_ids: list[int] = []
    for name in feature_names:
        group = feat_to_group.get(name, "Other")
        if group is None:
            feature_group_ids.append(-1)
            continue
        if group not in groups:
            groups.append(group)
        feature_group_ids.append(groups.index(group))

    shap_abs = np.abs(shap_values)
    group_matrix = np.zeros((len(feature_names), len(groups)), dtype=float)
    for feat_idx, group_idx in enumerate(feature_group_ids):
        if group_idx < 0:
            continue
        group_matrix[feat_idx, group_idx] = 1.0
    row_group_sums = shap_abs @ group_matrix

    per_patient = (
        pd.DataFrame(row_group_sums, columns=groups)
        .assign(patient_id=patient_ids)
        .groupby("patient_id")
        .mean()
    )

    rng = np.random.default_rng(random_state)
    point_estimate = per_patient.mean(axis=0)
    if normalize:
        total = point_estimate.sum()
        if total > 0:
            point_estimate = point_estimate / total

    ci: Dict[str, tuple[float | None, float | None]] = {}
    if n_boot > 0 and not per_patient.empty:
        samples: list[pd.Series] = []
        patient_ids_unique = per_patient.index.to_numpy()
        for _ in range(n_boot):
            sampled = rng.choice(
                patient_ids_unique, size=patient_ids_unique.size, replace=True
            )
            sample_df = per_patient.loc[sampled]
            agg = sample_df.mean(axis=0)
            if normalize:
                tot = agg.sum()
                if tot > 0:
                    agg = agg / tot
            samples.append(agg)
        boot_mat = pd.DataFrame(samples)
        for group in boot_mat.columns:
            low, high = np.percentile(boot_mat[group].dropna(), [2.5, 97.5])
            ci[group] = (float(low), float(high))
    return point_estimate.sort_values(ascending=False), ci


def _module_category(label: str) -> str:
    lower = label.lower()
    if "monitor" in lower or "count" in lower:
        return "Monitoring"
    return "Burden/biology/treatment"


def feature_abs_shap_with_ci(
    shap_path: Path,
    design_matrix: pd.DataFrame,
    idx_test: np.ndarray | None = None,
    normalize: bool = False,
    n_boot: int = 400,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, Dict[str, tuple[float | None, float | None]]]:
    data = np.load(shap_path, allow_pickle=True)
    shap_values = np.asarray(data["shap_values"], dtype=float)
    feature_names = [str(x) for x in data["feature_names"]]
    if "sample_idx" in data:
        sample_idx = np.asarray(data["sample_idx"], dtype=int)
    else:
        if idx_test is None or idx_test.size == 0:
            raise ValueError(
                "sample_idx missing from SHAP payload and idx_test not provided."
            )
        sample_idx = np.asarray(idx_test[: shap_values.shape[0]], dtype=int)
    if shap_values.shape[0] != sample_idx.shape[0]:
        raise ValueError("SHAP rows must align with sample_idx length.")
    if np.max(sample_idx) >= design_matrix.shape[0]:
        raise ValueError("sample_idx exceeds available design matrix rows.")

    shap_abs = np.abs(shap_values)
    global_idx = sample_idx
    patient_ids = design_matrix.iloc[global_idx][PATIENT_ID_COL].astype(str).to_numpy()

    per_patient = (
        pd.DataFrame(shap_abs, columns=feature_names)
        .assign(patient_id=patient_ids)
        .groupby("patient_id")
        .mean()
    )

    rng = np.random.default_rng(random_state)
    point_estimate = per_patient.mean(axis=0)
    if normalize:
        total = point_estimate.sum()
        if total > 0:
            point_estimate = point_estimate / total

    ci: Dict[str, tuple[float | None, float | None]] = {}
    if n_boot > 0 and not per_patient.empty:
        samples: list[pd.Series] = []
        patient_ids_unique = per_patient.index.to_numpy()
        for _ in range(n_boot):
            sampled = rng.choice(
                patient_ids_unique, size=patient_ids_unique.size, replace=True
            )
            sample_df = per_patient.loc[sampled]
            agg = sample_df.mean(axis=0)
            if normalize:
                tot = agg.sum()
                if tot > 0:
                    agg = agg / tot
            samples.append(agg)
        boot_mat = pd.DataFrame(samples)
        for feat in boot_mat.columns:
            low, high = np.percentile(boot_mat[feat].dropna(), [2.5, 97.5])
            ci[feat] = (float(low), float(high))

    return point_estimate.sort_values(ascending=False), ci


def plot_grouped_shap(
    contributions: pd.Series,
    ci: Dict[str, tuple[float | None, float | None]],
    output_path: Path,
) -> Path:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    groups = contributions.index.tolist()[::-1]
    values = contributions.loc[groups].to_numpy()
    errs: list[tuple[float, float]] = []
    category_colors = {
        "Monitoring": "#5b8ff9",
        "Burden/biology/treatment": "#174a5c",
    }
    colors: list[str] = []
    categories_seen: dict[str, bool] = {}
    for idx, name in enumerate(groups):
        lo, hi = ci.get(name, (None, None))
        if lo is None or hi is None:
            errs.append((0.0, 0.0))
        else:
            errs.append((values[idx] - lo, hi - values[idx]))
        cat = _module_category(name)
        colors.append(category_colors.get(cat, "#174a5c"))
        categories_seen[cat] = True
    err_array = np.array(errs).T if errs else None
    ax.barh(
        groups,
        values,
        xerr=err_array if np.any(err_array) else None,
        color=colors,
        ecolor="#444444",
        capsize=4,
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "Mean |SHAP| (normalized)" if contributions.sum() <= 1.01 else "Mean |SHAP|"
    )
    ax.set_title("Grouped SHAP contributions with 95% CI")
    legend_handles = []
    for cat, color in category_colors.items():
        if categories_seen.get(cat):
            legend_handles.append(
                plt.Line2D([0], [0], color=color, lw=8, label=cat, alpha=0.9)
            )
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    return output_path


def plot_top_features(
    contributions: pd.Series,
    ci: Mapping[str, tuple[float | None, float | None]],
    label_map: Mapping[str, str],
    output_path: Path,
    top_k: int = 10,
) -> Path:
    plt.rcParams.update(STYLE)
    top = contributions.sort_values(ascending=False).head(int(top_k))
    features = top.index.tolist()[::-1]
    values = top.to_numpy()[::-1]
    errs: list[tuple[float, float]] = []
    category_colors = {
        "Monitoring": "#5b8ff9",
        "Burden/biology/treatment": "#174a5c",
    }
    colors: list[str] = []
    categories_seen: dict[str, bool] = {}
    for idx, feat in enumerate(features):
        lo, hi = ci.get(feat, (None, None))
        if lo is None or hi is None:
            errs.append((0.0, 0.0))
        else:
            errs.append((values[idx] - lo, hi - values[idx]))
        module = feature_module(feat)
        cat = _module_category(module if module else feat)
        colors.append(category_colors.get(cat, "#174a5c"))
        categories_seen[cat] = True
    err_array = np.array(errs).T if errs else None

    display_labels = [label_map.get(f, f) for f in features]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        display_labels,
        values,
        xerr=err_array if np.any(err_array) else None,
        color=colors,
        ecolor="#444444",
        capsize=4,
    )
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("Top 10 baseline features by mean |SHAP| (patient-level)")
    ax.invert_yaxis()
    legend_handles = []
    for cat, color in category_colors.items():
        if categories_seen.get(cat):
            legend_handles.append(
                plt.Line2D([0], [0], color=color, lw=8, label=cat, alpha=0.9)
            )
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    return output_path


def _load_feature_labels(path: Path) -> Dict[str, str]:
    try:
        with open(path, "r") as fin:
            payload = json.load(fin)
        if isinstance(payload, dict):
            return {str(k): str(v) for k, v in payload.items()}
    except Exception:
        pass
    return {}


def _extract_shap_feature_names(shap_path: Path) -> list[str]:
    data = np.load(shap_path, allow_pickle=True)
    if "feature_names" not in data:
        return []
    return [str(x) for x in data["feature_names"]]


def top_baseline_feature_differences(
    risk_df: pd.DataFrame,
    strata_col: str,
    feature_names: Sequence[str],
    top_k: int = 5,
    label_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    label_map = label_map or {}
    df = risk_df[risk_df[strata_col].isin(STRATA_LABELS)].copy()
    if df.empty:
        return pd.DataFrame()

    cols = [name for name in feature_names if name in df.columns]
    if not cols:
        return pd.DataFrame()
    X = df[cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    tmp = pd.concat(
        [df[[strata_col]].reset_index(drop=True), X.reset_index(drop=True)], axis=1
    )

    means = tmp.groupby(strata_col)[cols].mean()
    stds = tmp.groupby(strata_col)[cols].std(ddof=0)
    counts = tmp.groupby(strata_col).size().reindex(STRATA_LABELS, fill_value=0)

    def _get(row: pd.DataFrame, key: str, feature: str) -> float:
        try:
            return float(row.loc[key, feature])
        except Exception:
            return float("nan")

    records: list[dict[str, object]] = []
    for feature in cols:
        low_mean = _get(means, "low", feature)
        mid_mean = _get(means, "mid", feature)
        high_mean = _get(means, "high", feature)
        low_std = _get(stds, "low", feature)
        high_std = _get(stds, "high", feature)
        pooled = float(np.sqrt((low_std * low_std + high_std * high_std) / 2.0))
        diff = high_mean - low_mean
        smd = diff / pooled if np.isfinite(pooled) and pooled > 0 else float("nan")
        abs_smd = abs(smd) if np.isfinite(smd) else float("nan")
        abs_diff = abs(diff) if np.isfinite(diff) else float("nan")
        rank = abs_smd if np.isfinite(abs_smd) else abs_diff
        if not np.isfinite(rank):
            continue
        records.append(
            {
                "feature": feature,
                "module": feature_module(feature),
                "label": label_map.get(feature, feature),
                "n_low": int(counts.get("low", 0)),
                "n_mid": int(counts.get("mid", 0)),
                "n_high": int(counts.get("high", 0)),
                "mean_low": low_mean,
                "mean_mid": mid_mean,
                "mean_high": high_mean,
                "diff_high_low": diff,
                "smd_high_low": smd,
                "rank": rank,
            }
        )
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records).sort_values("rank", ascending=False).head(int(top_k))
    return out.drop(columns=["rank"], errors="ignore")


def grouped_shap_long(
    shap_path: Path,
    design_matrix: pd.DataFrame,
) -> pd.DataFrame:
    data = np.load(shap_path, allow_pickle=True)
    shap_values = np.asarray(data["shap_values"], dtype=float)
    feature_names = [str(x) for x in data["feature_names"]]
    if "sample_idx" not in data:
        raise KeyError("'sample_idx' missing from SHAP payload.")
    sample_idx = np.asarray(data["sample_idx"], dtype=int)
    if shap_values.shape[0] != sample_idx.shape[0]:
        raise ValueError("SHAP rows must align with sample_idx length.")
    if np.max(sample_idx) >= design_matrix.shape[0]:
        raise ValueError("sample_idx exceeds available design matrix rows.")

    groups: list[str] = []
    feature_group_ids: list[int] = []
    for name in feature_names:
        group = feature_module(name)
        if group is None:
            feature_group_ids.append(-1)
            continue
        if group not in groups:
            groups.append(group)
        feature_group_ids.append(groups.index(group))

    shap_abs = np.abs(shap_values)
    group_matrix = np.zeros((len(feature_names), len(groups)), dtype=float)
    for feat_idx, group_idx in enumerate(feature_group_ids):
        if group_idx < 0:
            continue
        group_matrix[feat_idx, group_idx] = 1.0
    group_abs = shap_abs @ group_matrix
    denom = np.sum(group_abs, axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, np.nan)
    group_share = group_abs / denom

    global_idx = sample_idx
    patient_ids = design_matrix.iloc[global_idx][PATIENT_ID_COL].astype(str).to_numpy()

    long_rows: list[pd.DataFrame] = []
    for g_idx, group in enumerate(groups):
        long_rows.append(
            pd.DataFrame(
                {
                    "global_idx": global_idx,
                    "patient_id": patient_ids,
                    "group": group,
                    "shap_abs": group_abs[:, g_idx],
                    "shap_share": group_share[:, g_idx],
                }
            )
        )
    return pd.concat(long_rows, ignore_index=True)


def top_grouped_shap_differences(
    shap_long_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    strata_col: str,
    top_k: int = 5,
) -> pd.DataFrame:
    if shap_long_df.empty:
        return pd.DataFrame()
    meta = risk_df[["global_idx", strata_col]].dropna(subset=[strata_col])
    merged = shap_long_df.merge(meta, on="global_idx", how="inner")
    merged = merged[merged[strata_col].isin(STRATA_LABELS)]
    if merged.empty:
        return pd.DataFrame()

    agg = (
        merged.groupby([strata_col, "group"], observed=True)
        .agg(
            mean_abs=("shap_abs", "mean"),
            std_abs=("shap_abs", lambda s: s.std(ddof=0)),
            mean_share=("shap_share", "mean"),
            std_share=("shap_share", lambda s: s.std(ddof=0)),
        )
        .reset_index()
    )

    abs_mean = agg.pivot(index="group", columns=strata_col, values="mean_abs").reindex(
        columns=STRATA_LABELS
    )
    abs_std = agg.pivot(index="group", columns=strata_col, values="std_abs").reindex(
        columns=STRATA_LABELS
    )
    share_mean = agg.pivot(
        index="group", columns=strata_col, values="mean_share"
    ).reindex(columns=STRATA_LABELS)
    share_std = agg.pivot(
        index="group", columns=strata_col, values="std_share"
    ).reindex(columns=STRATA_LABELS)

    out = pd.DataFrame(index=abs_mean.index)
    out["mean_abs_low"] = abs_mean["low"]
    out["mean_abs_mid"] = abs_mean["mid"]
    out["mean_abs_high"] = abs_mean["high"]
    out["diff_abs_high_low"] = out["mean_abs_high"] - out["mean_abs_low"]

    pooled_abs = np.sqrt(
        (
            abs_std["low"].to_numpy(dtype=float) ** 2
            + abs_std["high"].to_numpy(dtype=float) ** 2
        )
        / 2.0
    )
    diff_abs = out["diff_abs_high_low"].to_numpy(dtype=float)
    out["smd_abs_high_low"] = np.divide(
        diff_abs,
        pooled_abs,
        out=np.full_like(diff_abs, np.nan, dtype=float),
        where=np.isfinite(pooled_abs) & (pooled_abs > 0),
    )

    out["mean_share_low"] = share_mean["low"]
    out["mean_share_mid"] = share_mean["mid"]
    out["mean_share_high"] = share_mean["high"]
    out["diff_share_high_low"] = out["mean_share_high"] - out["mean_share_low"]

    pooled_share = np.sqrt(
        (
            share_std["low"].to_numpy(dtype=float) ** 2
            + share_std["high"].to_numpy(dtype=float) ** 2
        )
        / 2.0
    )
    diff_share = out["diff_share_high_low"].to_numpy(dtype=float)
    out["smd_share_high_low"] = np.divide(
        diff_share,
        pooled_share,
        out=np.full_like(diff_share, np.nan, dtype=float),
        where=np.isfinite(pooled_share) & (pooled_share > 0),
    )

    # Alias for consumers who want to report SMD consistently.
    out["smd_high_low"] = out["smd_abs_high_low"]

    out = out.reset_index().rename(columns={"index": "group"})
    out["rank"] = out["smd_high_low"].abs()
    out = out.sort_values("rank", ascending=False).head(int(top_k))
    return out.drop(columns=["rank"], errors="ignore")


def generate_risk_figures(
    design_matrix_path: Path,
    surv_npz_path: Path,
    shap_npz_path: Path,
    features_dict_path: Path,
    feature_labels_path: Path | None,
    output_dir: Path,
    horizon_days: float = 365.0,
    n_boot: int = 400,
    top_k_diffs: int = 10,
    use_manuscript_groups: bool = True,
) -> Dict[str, Any]:
    design_matrix = _load_design_matrix(Path(design_matrix_path))
    time_grid, surv, idx_from_npz = _load_survival_predictions(Path(surv_npz_path))
    if idx_from_npz is None:
        raise ValueError("idx_test is required inside survival npz.")
    idx_test = np.asarray(idx_from_npz, dtype=int)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: Dict[str, Path] = {}
    tables: Dict[str, Path] = {}

    strat_fig, strat_summary, strat_details = plot_tertile_stratified_survival(
        indices=idx_test,
        time_grid=time_grid,
        surv_np=surv,
        design_matrix=design_matrix,
        output_path=output_dir / "stratified_survival.png",
        horizon_days=horizon_days,
        rmst_tau=None,
        n_boot=n_boot,
        random_state=RANDOM_STATE,
    )
    summary_csv = output_dir / "risk_summary.csv"
    strat_summary.to_csv(summary_csv, index=False)
    figures["stratified_survival"] = strat_fig
    tables["risk_summary"] = summary_csv

    label_map = (
        _load_feature_labels(Path(feature_labels_path))
        if feature_labels_path is not None
        else {}
    )
    contributions, ci = grouped_shap_contributions(
        shap_path=Path(shap_npz_path),
        features_dict_path=Path(features_dict_path),
        design_matrix=design_matrix,
        idx_test=idx_test,
        normalize=False,
        n_boot=n_boot,
        use_manuscript_groups=use_manuscript_groups,
    )
    figures["grouped_shap"] = plot_grouped_shap(
        contributions=contributions,
        ci=ci,
        output_path=output_dir / "fig_c1_grouped_shap.png",
    )
    top_feat_contrib, top_feat_ci = feature_abs_shap_with_ci(
        shap_path=Path(shap_npz_path),
        design_matrix=design_matrix,
        idx_test=idx_test,
        normalize=False,
        n_boot=n_boot,
    )
    figures["top_features"] = plot_top_features(
        contributions=top_feat_contrib,
        ci=top_feat_ci,
        label_map=label_map,
        output_path=output_dir / "fig_c0_top_features.png",
        top_k=top_k_diffs,
    )

    return {
        "risk_summary": strat_summary,
        "risk_details": strat_details,
        "figures": figures,
        "tables": tables,
    }
