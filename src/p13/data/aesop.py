"""AESOP → unified JSONL.  **Stub – owner: Parsa.**

AESOP stories are 3-panel clip-art scenes with a text per panel. Panels may need to be
rendered to images from the scene definitions if the release does not ship rasters.

To implement :func:`prepare`:

1. Read the AESOP release from ``data/raw/aesop/``; document files in ``data/README.md``.
2. Obtain one image per panel (3 per story), checked and resized with :mod:`p13.data.images`.
3. ``story`` = the panel texts joined in order; keep per-panel texts in ``dis`` if useful.
4. Build :class:`p13.common.schema.StoryRecord` with ``dataset="aesop"``.
5. Write ``data/processed/aesop_<split>.jsonl`` and the dev-200 ID list.
6. Add ``tests/test_aesop.py`` with a tiny fake release.
"""

from __future__ import annotations

from pathlib import Path

from p13.common.config import ExperimentConfig


def prepare(cfg: ExperimentConfig, split: str) -> Path:
    raise NotImplementedError("AESOP loader not implemented yet – see module docstring (Parsa)")
