from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

DEFAULT_MODELS = ("coxph", "deephit", "deepsurv", "gbsa", "rsf")
MIN_CONFIG_COUNT = 3


def build_ablation_configs(
    results_root: Path,
    models: Sequence[str],
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
            name for name, n in counts.items() if n >= int(MIN_CONFIG_COUNT)
        )
        if not configs:
            raise ValueError(
                f"No ablation configs found for model {model} with count >= {MIN_CONFIG_COUNT}."
            )
        out[model] = configs
    return out


def main() -> None:
    out_path: Path = Path("data/ablation_configs.json")
    configs = build_ablation_configs(
        results_root=Path("results"),
        models=DEFAULT_MODELS,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
