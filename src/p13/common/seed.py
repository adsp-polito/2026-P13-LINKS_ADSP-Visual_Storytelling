"""Reproducibility: one call seeds every RNG we use (default 42)."""

from __future__ import annotations

import os
import random

import numpy as np

from p13.common.config import DEFAULT_SEED


def set_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = False) -> int:
    """Seed ``random``, ``numpy`` and (if installed) ``torch`` / CUDA.

    ``deterministic=True`` additionally asks torch for deterministic kernels (slower).
    Returns the seed so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    return seed
