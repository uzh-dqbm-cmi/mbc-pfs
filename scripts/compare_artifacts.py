from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Diff:
    path: str
    message: str


def _iter_files(root: Path, exts: set[str]) -> Iterator[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in exts:
                yield p


def _truncate_float(x: float, decimals: int) -> float:
    if not math.isfinite(x):
        return x
    scale = 10.0**decimals
    return float(math.trunc(float(x) * scale) / scale)


def _truncate_array(arr: np.ndarray, decimals: int) -> np.ndarray:
    scale = 10.0**decimals
    return np.trunc(arr * scale) / scale


def _normalize_json(obj: Any, *, decimals: int) -> Any:
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return _truncate_float(obj, decimals)
    if isinstance(obj, list):
        return [_normalize_json(v, decimals=decimals) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_normalize_json(v, decimals=decimals) for v in obj)
    if isinstance(obj, dict):
        # normalize keys to strings for stable comparisons; sort not required for equality but helps dumps.
        return {
            str(k): _normalize_json(v, decimals=decimals)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    # numpy scalar types
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return _normalize_json(obj.item(), decimals=decimals)
    return str(obj)


def _diff_json(a: Any, b: Any, *, decimals: int, prefix: str = "") -> list[Diff]:
    diffs: list[Diff] = []

    def walk(x: Any, y: Any, path: str):
        if type(x) != type(y):
            diffs.append(Diff(path, f"type mismatch: {type(x).__name__} vs {type(y).__name__}"))
            return

        if isinstance(x, dict):
            xk = set(x.keys())
            yk = set(y.keys())
            for k in sorted(xk - yk):
                diffs.append(Diff(f"{path}.{k}" if path else str(k), "missing in B"))
            for k in sorted(yk - xk):
                diffs.append(Diff(f"{path}.{k}" if path else str(k), "missing in A"))
            for k in sorted(xk & yk):
                walk(x[k], y[k], f"{path}.{k}" if path else str(k))
            return

        if isinstance(x, list):
            if len(x) != len(y):
                diffs.append(Diff(path, f"list length mismatch: {len(x)} vs {len(y)}"))
                return
            for i, (xi, yi) in enumerate(zip(x, y)):
                walk(xi, yi, f"{path}[{i}]")
            return

        if isinstance(x, float):
            if _truncate_float(x, decimals) != _truncate_float(y, decimals):
                diffs.append(
                    Diff(
                        path,
                        f"value mismatch: {x} vs {y} (first {decimals} decimals differ)",
                    )
                )
            return

        if x != y:
            diffs.append(Diff(path, f"value mismatch: {x!r} vs {y!r}"))

    walk(_normalize_json(a, decimals=decimals), _normalize_json(b, decimals=decimals), prefix)
    return diffs


def _is_numeric_array(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.number)


def _nan_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    if not np.array_equal(a_nan, b_nan):
        return False
    return True


def _compare_arrays(
    a: np.ndarray,
    b: np.ndarray,
    *,
    decimals: int,
) -> tuple[bool, str | None]:
    if a.shape != b.shape:
        return False, f"shape mismatch {a.shape} vs {b.shape}"
    if a.dtype != b.dtype:
        # allow safe comparison between float dtypes after rounding
        if _is_numeric_array(a) and _is_numeric_array(b):
            a = a.astype(float)
            b = b.astype(float)
        else:
            return False, f"dtype mismatch {a.dtype} vs {b.dtype}"
    if _is_numeric_array(a) and _is_numeric_array(b):
        af = a.astype(float, copy=False)
        bf = b.astype(float, copy=False)
        if not _nan_equal(af, bf):
            return False, "NaN pattern mismatch"
        mask = ~np.isnan(af)
        if mask.any():
            ar = _truncate_array(af[mask], decimals=decimals)
            br = _truncate_array(bf[mask], decimals=decimals)
            if not np.array_equal(ar, br):
                idx = int(np.flatnonzero(ar != br)[0])
                return (
                    False,
                    f"value mismatch at flat idx {idx}: {af[mask][idx]} vs {bf[mask][idx]} "
                    f"(first {decimals} decimals differ)",
                )
        return True, None
    return (np.array_equal(a, b), None if np.array_equal(a, b) else "array mismatch")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_npz(path: Path) -> Mapping[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: np.asarray(data[k]) for k in data.files}


def compare_files(a_path: Path, b_path: Path, *, decimals: int) -> list[Diff]:
    if a_path.suffix.lower() != b_path.suffix.lower():
        return [Diff("", f"extension mismatch: {a_path.suffix} vs {b_path.suffix}")]

    if a_path.suffix.lower() == ".json":
        a_obj = _load_json(a_path)
        b_obj = _load_json(b_path)
        return _diff_json(a_obj, b_obj, decimals=decimals)

    if a_path.suffix.lower() == ".npz":
        a_npz = _load_npz(a_path)
        b_npz = _load_npz(b_path)
        diffs: list[Diff] = []
        a_keys = set(a_npz.keys())
        b_keys = set(b_npz.keys())
        for k in sorted(a_keys - b_keys):
            diffs.append(Diff(k, "missing array in B"))
        for k in sorted(b_keys - a_keys):
            diffs.append(Diff(k, "missing array in A"))
        for k in sorted(a_keys & b_keys):
            ok, msg = _compare_arrays(a_npz[k], b_npz[k], decimals=decimals)
            if not ok:
                diffs.append(Diff(k, msg or "array mismatch"))
        return diffs

    return [Diff("", f"unsupported extension: {a_path.suffix}")]


def compare_trees(a_root: Path, b_root: Path, *, decimals: int, exts: set[str]) -> int:
    a_root = a_root.resolve()
    b_root = b_root.resolve()
    a_files = {p.relative_to(a_root): p for p in _iter_files(a_root, exts)}
    b_files = {p.relative_to(b_root): p for p in _iter_files(b_root, exts)}

    all_rel = sorted(set(a_files) | set(b_files))
    missing: list[str] = []
    different: list[str] = []

    for rel in all_rel:
        a_path = a_files.get(rel)
        b_path = b_files.get(rel)
        if a_path is None:
            missing.append(f"{rel} (missing in A)")
            continue
        if b_path is None:
            missing.append(f"{rel} (missing in B)")
            continue
        diffs = compare_files(a_path, b_path, decimals=decimals)
        if diffs:
            different.append(str(rel))
            print(f"\n=== DIFF {rel} ===")
            for d in diffs[:50]:
                print(f"- {d.path}: {d.message}")
            if len(diffs) > 50:
                print(f"... ({len(diffs) - 50} more)")

    if missing:
        print("\n=== MISSING FILES ===")
        for line in missing[:200]:
            print(f"- {line}")
        if len(missing) > 200:
            print(f"... ({len(missing) - 200} more)")

    print("\n=== SUMMARY ===")
    print(f"- compared: {len(all_rel)}")
    print(f"- different: {len(different)}")
    print(f"- missing: {len(missing)}")

    return 1 if (missing or different) else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare .json and .npz artifacts across two directories/files "
            "(order-insensitive JSON; compare first N decimal digits)."
        )
    )
    parser.add_argument("a", type=Path, help="Path A (directory root or file)")
    parser.add_argument("b", type=Path, help="Path B (directory root or file)")
    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help=(
            "Compare floats by matching the first N digits after the decimal point "
            "(truncation, default: 6)."
        ),
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=[".json", ".npz"],
        help="Extensions to compare (repeatable). Default: .json and .npz",
    )
    args = parser.parse_args(argv)
    if args.decimals < 0:
        raise SystemExit("--decimals must be >= 0")

    exts = {e if e.startswith(".") else f".{e}" for e in args.ext}

    a = args.a
    b = args.b
    if a.is_file() and b.is_file():
        diffs = compare_files(a, b, decimals=args.decimals)
        if diffs:
            print(f"=== DIFF {a.name} ===")
            for d in diffs:
                print(f"- {d.path}: {d.message}")
            return 1
        print("No differences found.")
        return 0

    if a.is_dir() and b.is_dir():
        return compare_trees(a, b, decimals=args.decimals, exts=exts)

    raise SystemExit("Both inputs must be files or both must be directories.")


if __name__ == "__main__":
    raise SystemExit(main())
