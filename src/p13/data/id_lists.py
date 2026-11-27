"""Shared helpers for the committed id lists in ``data/splits/``.  **Owner: Parsa.**

Used by the dataset loaders (VWP, VIST) to pick one story per group (scene, album,
image sequence) and to write each committed list once, never overwriting it.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path

from p13.common.subset import read_ids, write_ids

log = logging.getLogger(__name__)


def one_per_group(ids: set[str], group_of: dict[str, str], seed: int) -> list[str]:
    """One random kept story per group (groups and ids sorted first, so it is deterministic)."""
    by_group: dict[str, list[str]] = defaultdict(list)
    for i in sorted(ids):
        by_group[group_of[i]].append(i)
    rng = random.Random(seed)
    return [rng.choice(by_group[g]) for g in sorted(by_group)]


def ensure_id_list(path: Path, ids: list[str], built: set[str], drops: dict[str, int]) -> None:
    """Write the committed id list ``path`` once; never overwrite an existing one.

    Raises if an existing list names stories that were not built here, because eval would
    then silently run on fewer stories than the committed list. Writes nothing if no story
    was built or if images are missing (the download is incomplete).
    """
    if path.exists():
        absent = [i for i in read_ids(path) if i not in built]
        if absent:
            raise RuntimeError(
                f"{path} lists {len(absent)} ids that were not built here (e.g. {absent[0]}); "
                f"check the images (drop counts: {drops})"
            )
        return
    if not built:
        log.warning("not writing %s: no stories were built", path)
        return
    if drops["missing_image"]:
        log.warning(
            "not writing %s: %d stories lack images. Download all images first.",
            path,
            drops["missing_image"],
        )
        return
    write_ids(path, ids)
    log.info("wrote %s (%d ids)", path, len(ids))
