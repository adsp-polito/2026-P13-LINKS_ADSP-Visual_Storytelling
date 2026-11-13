"""VIST → unified JSONL.  **Skeleton – owner: Parsa.**

Expected raw layout (see ``data/README.md``)::

    data/raw/vist/sis/{train,val,test}.story-in-sequence.json
    data/raw/vist/dii/{train,val,test}.description-in-isolation.json   (optional)
    data/raw/vist/dis/{train,val,test}.description-in-sequence.json    (optional)
    data/raw/vist/images/{train,val,test}/<photo_flickr_id>.<ext>

Pipeline for :func:`prepare` (only step 4 is implemented):

1. Group SIS annotations by ``story_id``; order photos by ``worker_arranged_photo_order``.
2. Drop a whole sequence if any image is missing or broken (``images.is_valid_image``).
3. Resize images to long side ≤ ``cfg.data.max_image_side`` (``images.resize_long_side``)
   into ``data/images/vist/<split>/``.
4. Rewrite placeholders ``[male]`` / ``[female]`` / ``[location]`` / ``[organization]``
   (:func:`rewrite_placeholders`).
5. Attach per-photo DII / DIS text if those files exist.
6. Write ``data/processed/vist_<split>.jsonl`` (``write_jsonl``) and, for val/test, the
   dev-200 id list (``p13.common.subset``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from p13.common.config import ExperimentConfig
from p13.common.schema import StoryRecord

# Edit here if the team decides on different wording.
PLACEHOLDERS: dict[str, str] = {
    "[male]": "a man",
    "[female]": "a woman",
    "[location]": "a place",
    "[organization]": "an organization",
}
_PLACEHOLDER_RE = re.compile(r"\[(male|female|location|organization)\]", re.IGNORECASE)


def rewrite_placeholders(text: str) -> str:
    """Replace VIST anonymisation tokens with generic noun phrases.

    Later mentions of the same placeholder use "the" instead of "a/an":
    ``"[male] went home . [male] slept"`` → ``"a man went home . the man slept"``.
    """
    seen: set[str] = set()

    def sub(m: re.Match[str]) -> str:
        key = f"[{m.group(1).lower()}]"
        phrase = PLACEHOLDERS[key]
        if key in seen:
            phrase = "the " + phrase.split(" ", 1)[1]
        seen.add(key)
        return phrase

    return _PLACEHOLDER_RE.sub(sub, text)


@dataclass(frozen=True)
class RawStory:
    story_id: str
    photo_ids: list[str]
    sentences: list[str]


def group_sis(sis: dict) -> list[RawStory]:
    """Step 1: flat SIS annotation list → ordered stories."""
    raise NotImplementedError("TODO (Parsa)")


def per_photo_text(path: Path) -> dict[str, str]:
    """Step 5: ``photo_flickr_id`` → description from a DII or DIS file ({} if absent)."""
    raise NotImplementedError("TODO (Parsa)")


def build_records(
    stories: list[RawStory],
    *,
    split: str,
    raw_image_dir: Path,
    out_image_dir: Path,
    max_image_side: int,
) -> tuple[list[StoryRecord], dict[str, int]]:
    """Steps 2–5: return ``(records, drop_counts)``; log how many stories were dropped and why."""
    raise NotImplementedError("TODO (Parsa)")


def prepare(cfg: ExperimentConfig, split: str) -> Path:
    """Run steps 1–6 for one split; return the written JSONL path."""
    raise NotImplementedError("TODO (Parsa)")
