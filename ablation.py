#!/usr/bin/env python3
"""Run outer-fold modality ablations using existing cv_splits/* files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

from config import MODALITY_GROUPS


def _run(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("==>", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(list(cmd), check=True)


def _build_cmd(args: argparse.Namespace, *, mode: str, group: str | None = None) -> list[str]:
    cmd = [
        sys.executable,
        f"{args.runner}_nested_cv.py",
        "--models",
        args.models,
    ]
    if mode == "exclude":
        cmd.extend(["--exclude-modality", group])
    elif mode == "only":
        cmd.extend(["--only-modality", group])
    elif mode == "drop-age":
        cmd.append("--drop-age")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if args.max_workers is not None and args.runner == "cpu":
        cmd.extend(["--max-workers", str(args.max_workers)])
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run outer-fold ablations for each modality group in config.MODALITY_GROUPS, "
            "reusing existing cv_splits/*.csv and the single config in data/ablation_configs.json."
        )
    )
    ap.add_argument(
        "--runner",
        required=True,
        # fixed string of cpu or dl
        type=str,
        choices=["cpu", "dl"],
    )
    ap.add_argument(
        "--models",
        default="dryrun",
        help="Comma-separated model list passed to cpu_nested_cv.py.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    ap.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Max concurrent ablation runs (default: 1).",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="If set and --runner=cpu, pass through to cpu_nested_cv.py.",
    )
    args = ap.parse_args()

    if args.max_parallel < 1:
        ap.error("--max-parallel must be >= 1")

    commands: list[tuple[str, list[str]]] = []
    for group in MODALITY_GROUPS.keys():
        commands.append((f"exclude:{group}", _build_cmd(args, mode="exclude", group=group)))
        commands.append((f"only:{group}", _build_cmd(args, mode="only", group=group)))
    commands.append(("drop-age", _build_cmd(args, mode="drop-age")))

    failures = 0
    if args.max_parallel == 1:
        for label, cmd in commands:
            try:
                _run(cmd, dry_run=args.dry_run)
            except subprocess.CalledProcessError as e:
                print(f"Error running ablation '{label}': {e}")
                failures += 1
    else:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
            futures = {
                executor.submit(_run, cmd, dry_run=args.dry_run): label
                for label, cmd in commands
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    future.result()
                except subprocess.CalledProcessError as e:
                    print(f"Error running ablation '{label}': {e}")
                    failures += 1
                except Exception as e:
                    print(f"Error running ablation '{label}': {e}")
                    failures += 1
    if failures:
        print(f"Completed with {failures} failed ablation(s).")


if __name__ == "__main__":
    main()
