from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch


def _seed_reproducibly(seed: int) -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="The NumPy global RNG was seeded.*",
    )
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    if hasattr(torch, "mps"):
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
