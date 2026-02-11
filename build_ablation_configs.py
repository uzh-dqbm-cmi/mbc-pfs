from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

DEFAULT_MODELS = ("coxph", "deephit", "deepsurv", "gbsa", "rsf")


def build_ablation_configs(
    results_root: Path,
    models: Sequence[str] = DEFAULT_MODELS,
    min_config_count: int = 2,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for model in models:
        eval_path = results_root / model / "eval_metrics.json"
        payload = json.load(open(eval_path))
        outer = payload.get("outer_folds", {})
        counts = Counter(
            Path(item["results_path"]).name
            for item in outer.values()
            if isinstance(item, dict) and item.get("results_path")
        )
        configs = sorted(
            name for name, n in counts.items() if n >= int(min_config_count)
        )
        if not configs:
            raise ValueError(
                f"No ablation configs found for model {model} with count >= {min_config_count}."
            )
        out[model] = configs
    return out


def write_ablation_configs(
    out_path: Path = Path("data/ablation_configs.json"),
    results_root: Path = Path("results"),
    models: Sequence[str] = DEFAULT_MODELS,
    min_config_count: int = 2,
) -> dict[str, list[str]]:
    configs = build_ablation_configs(
        results_root=results_root,
        models=models,
        min_config_count=min_config_count,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(configs, f, indent=2)
    return configs


def main() -> None:
    models = DEFAULT_MODELS
    write_ablation_configs(
        out_path=Path("data/ablation_configs.json"),
        results_root=Path("results"),
        models=models,
        min_config_count=2,
    )
    print(f"Wrote {Path('data/ablation_configs.json')}")


if __name__ == "__main__":
    main()
