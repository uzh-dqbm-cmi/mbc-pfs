from __future__ import annotations

from pathlib import Path

from config import LINE_START_WINDOW_DAYS


def data_dir(window_days: int = LINE_START_WINDOW_DAYS) -> Path:
    if window_days == LINE_START_WINDOW_DAYS:
        return Path("data")
    return Path("data") / f"window_{window_days}d"


def window_results_root(window_days: int) -> Path:
    return Path("results_window") / f"{window_days}d"
