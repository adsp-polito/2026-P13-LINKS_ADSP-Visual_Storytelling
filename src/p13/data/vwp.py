"""VWP (Visual Writing Prompts) → unified JSONL.  **Stub – owner: Parsa.**

VWP pairs sequences of 5–10 movie shots with crowd-written stories. Characters are grounded
to the shots (character names come from the films). For our grounding rules, names should be
replaced with generic noun phrases the same way VIST placeholders are (see
:func:`p13.data.vist.rewrite_placeholders`).

To implement :func:`prepare`:

1. Read the VWP release (CSV/JSON) from ``data/raw/vwp/``; document the exact files in
   ``data/README.md``.
2. Map each story to its ordered shot images; download/locate shots into ``data/raw/vwp/images``.
3. Reuse :mod:`p13.data.images` (``is_valid_image``, ``resize_long_side``) – same 448 px policy.
4. Build :class:`p13.common.schema.StoryRecord` with ``dataset="vwp"``; ``dis``/``dii``
   stay empty unless VWP provides per-shot descriptions.
5. Write ``data/processed/vwp_<split>.jsonl`` and the dev-200 ID list (see ``vist.prepare``).
6. Add ``tests/test_vwp.py`` with a tiny fake release, mirroring ``tests/test_vist.py``.
"""

from __future__ import annotations

from pathlib import Path

from p13.common.config import ExperimentConfig


def prepare(cfg: ExperimentConfig, split: str) -> Path:
    raise NotImplementedError("VWP loader not implemented yet – see module docstring (Parsa)")
