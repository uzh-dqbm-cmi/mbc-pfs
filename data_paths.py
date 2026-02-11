"""Helpers for constructing suffixed data/result artifact paths."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_DATA_SUFFIX = ""


def set_data_suffix(suffix: Optional[str]) -> None:
    global _DATA_SUFFIX
    _DATA_SUFFIX = (suffix or "").strip()


def get_data_suffix() -> str:
    return _DATA_SUFFIX


def _apply(stem: str, explicit_suffix: Optional[str]) -> str:
    suffix = (explicit_suffix if explicit_suffix is not None else _DATA_SUFFIX).strip()
    if not suffix:
        return stem
    return f"{stem}_{suffix}"


def data_csv(stem: str, suffix: Optional[str] = None) -> Path:
    return Path("data") / f"{_apply(stem, suffix)}.csv"


def data_json(stem: str, suffix: Optional[str] = None) -> Path:
    return Path("data") / f"{_apply(stem, suffix)}.json"


def results_root(base: str = "results", suffix: Optional[str] = None) -> str:
    """Return the directory that should hold model outputs."""

    base_clean = (base or "results").strip()
    suffix_clean = (suffix or "").strip()
    if not suffix_clean:
        return base_clean
    if base_clean.endswith(f"_{suffix_clean}"):
        return base_clean
    return f"{base_clean}_{suffix_clean}"
