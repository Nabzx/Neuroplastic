"""Deterministic seeding across the scientific-Python + PyTorch stack.

Reproducibility is a first-class concern for this project: every reported
result must be traceable to a ``(config, seed)`` pair. This helper centralises
seeding so experiments do not have to re-implement it.
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    Parameters
    ----------
    seed:
        The seed to apply everywhere.
    deterministic:
        If ``True``, additionally request deterministic CUDA/cuDNN kernels.
        This can slow training down and is intended for final, publishable
        runs rather than day-to-day development.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch is optional at import time so that lightweight tooling (config
        # validation, analysis of saved runs) works without it installed.
        pass


__all__ = ["set_global_seed"]
