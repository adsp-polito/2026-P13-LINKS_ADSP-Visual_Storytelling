"""Fixed story subsets: committed id lists in ``data/splits/``.

A list is named ``<dataset>_<split>_<tag>.txt``. The dev lists (tag ``dev<N>``, default
200 stories, seed 42) are written once by ``p13-prepare``; other tags (``1010``, ``791``)
name fixed evaluation lists. Lists are committed, so every teammate uses exactly the same
stories. :func:`load_subset_records` is the one place that turns ``data.subset`` into
records, for generation and evaluation alike.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

from p13.common.config import DEFAULT_SEED, SUBSET_TAG_RE, ExperimentConfig
from p13.common.jsonl import read_jsonl
from p13.common.schema import StoryRecord


def sample_ids(ids: Iterable[str], n: int = 200, seed: int = DEFAULT_SEED) -> list[str]:
    """Deterministic sample of ``n`` ids (sorted first, so input order does not matter)."""
    pool = sorted(set(ids))
    if n >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n))


def subset_path(splits_dir: str | Path, dataset: str, split: str, n: int = 200) -> Path:
    return Path(splits_dir) / f"{dataset}_{split}_dev{n}.txt"


def subset_file(splits_dir: str | Path, dataset: str, split: str, tag: str) -> Path:
    """``<splits_dir>/<dataset>_<split>_<tag>.txt``; ``tag`` is e.g. ``dev200`` or ``1010``."""
    if not SUBSET_TAG_RE.fullmatch(tag):
        raise ValueError(f"bad subset tag {tag!r}: use a-z and 0-9, not starting with 0")
    return Path(splits_dir) / f"{dataset}_{split}_{tag}.txt"


def load_subset_records(cfg: ExperimentConfig, dataset: str) -> list[StoryRecord]:
    """Stories of ``dataset`` / ``cfg.data.split`` selected by ``cfg.data.subset``.

    ``"full"`` returns every record of the JSONL file. Any other tag returns the records
    listed in :func:`subset_file`, in list order. Raises if the JSONL or the list file is
    missing, or if the list names an id that is not in the JSONL file.
    """
    split = cfg.data.split
    jsonl_path = Path(cfg.data.jsonl_dir) / f"{dataset}_{split}.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"{jsonl_path} not found; run p13-prepare or unpack the processed JSONL first"
        )
    records = read_jsonl(jsonl_path, StoryRecord)
    if cfg.data.subset == "full":
        return records
    ids_path = subset_file(cfg.data.splits_dir, dataset, split, cfg.data.subset)
    if not ids_path.is_file():
        raise FileNotFoundError(
            f"{ids_path} not found; run p13-prepare first or use --set data.subset=full"
        )
    by_id = {r.id: r for r in records}
    want = read_ids(ids_path)
    missing = [i for i in want if i not in by_id]
    if missing:
        raise RuntimeError(
            f"{ids_path} lists {len(missing)} id(s) missing from {jsonl_path} (e.g. {missing[0]})"
        )
    return [by_id[i] for i in want]


def groups_path(jsonl_dir: str | Path, dataset: str, split: str) -> Path:
    """``<jsonl_dir>/<dataset>_<split>_groups.tsv``: story id -> group (VWP scene, VIST album)."""
    return Path(jsonl_dir) / f"{dataset}_{split}_groups.tsv"


def write_groups(path: str | Path, groups: dict[str, str]) -> None:
    """Write ``id<TAB>group`` lines (after an ``id<TAB>group`` header), in the given order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id\tgroup\n", *(f"{i}\t{g}\n" for i, g in groups.items())]
    path.write_text("".join(lines), encoding="utf-8")


def read_groups(path: str | Path) -> dict[str, str]:
    """Inverse of :func:`write_groups`."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "id\tgroup":
        raise ValueError(f"{path}: expected an 'id<TAB>group' header")
    return dict(line.split("\t", 1) for line in lines[1:] if line)


def write_ids(path: str | Path, ids: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")


def read_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
