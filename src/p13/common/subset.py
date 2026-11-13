"""Fixed dev subsets (default: 200 sequences per dataset, seed 42).

The ID list is written once to ``data/splits/<dataset>_<split>_dev<N>.txt`` and committed,
so every teammate evaluates on exactly the same sequences.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

from p13.common.config import DEFAULT_SEED


def sample_ids(ids: Iterable[str], n: int = 200, seed: int = DEFAULT_SEED) -> list[str]:
    """Deterministic sample of ``n`` ids (sorted first, so input order does not matter)."""
    pool = sorted(set(ids))
    if n >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n))


def subset_path(splits_dir: str | Path, dataset: str, split: str, n: int = 200) -> Path:
    return Path(splits_dir) / f"{dataset}_{split}_dev{n}.txt"


def write_ids(path: str | Path, ids: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")


def read_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
