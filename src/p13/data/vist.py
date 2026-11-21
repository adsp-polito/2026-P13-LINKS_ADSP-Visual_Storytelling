"""VIST → unified JSONL.  **Owner: Parsa.**

Expected raw layout (see ``data/README.md``)::

    data/raw/vist/sis/{train,val,test}.story-in-sequence.json
    data/raw/vist/dii/{train,val,test}.description-in-isolation.json   (optional)
    data/raw/vist/dis/{train,val,test}.description-in-sequence.json    (optional)
    data/raw/vist/images/{train,val,test}/<photo_flickr_id>.<ext>

Pipeline for :func:`prepare`:

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

import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from p13.common.config import ExperimentConfig
from p13.common.jsonl import write_jsonl
from p13.common.schema import MAX_IMAGES, MIN_IMAGES, StoryRecord
from p13.common.subset import read_ids, sample_ids, subset_path, write_ids
from p13.data.images import find_image, is_valid_image, resize_long_side

log = logging.getLogger(__name__)

# Edit here if the team decides on different wording.
PLACEHOLDERS: dict[str, str] = {
    "[male]": "a man",
    "[female]": "a woman",
    "[location]": "a place",
    "[organization]": "an organization",
}
_PLACEHOLDER_RE = re.compile(r"\[(male|female|location|organization)\]", re.IGNORECASE)

# Annotation field used as text. "text" is lower-cased and tokenised;
# "original_text" keeps the raw casing. Edit here if the team decides otherwise.
TEXT_FIELD = "text"

# Why a whole story can be dropped (keys of the ``drop_counts`` dict).
DROP_REASONS = ("bad_length", "empty_story", "empty_sentence", "missing_image", "broken_image")


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


def _flatten(annotations: list) -> Iterator[dict]:
    """VIST wraps each annotation in a one-element list; accept wrapped and flat items."""
    for item in annotations:
        if isinstance(item, dict):
            yield item
        else:
            yield from item


def group_sis(sis: dict) -> list[RawStory]:
    """Step 1: flat SIS annotation list → ordered stories.

    Stories keep the order in which their ``story_id`` first appears in the file;
    photos inside a story are sorted by ``worker_arranged_photo_order``.
    """
    by_story: dict[str, list[dict]] = defaultdict(list)
    for ann in _flatten(sis["annotations"]):
        by_story[str(ann["story_id"])].append(ann)

    stories = []
    for story_id, anns in by_story.items():
        anns.sort(key=lambda a: int(a["worker_arranged_photo_order"]))
        stories.append(
            RawStory(
                story_id=story_id,
                photo_ids=[str(a["photo_flickr_id"]) for a in anns],
                sentences=[a[TEXT_FIELD].strip() for a in anns],
            )
        )
    return stories


def per_photo_text(path: Path) -> dict[str, str]:
    """Step 5: ``photo_flickr_id`` → description from a DII or DIS file ({} if absent).

    If a photo has several descriptions, the first one in the file is kept.
    """
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    texts: dict[str, str] = {}
    for ann in _flatten(data["annotations"]):
        texts.setdefault(str(ann["photo_flickr_id"]), ann[TEXT_FIELD].strip())
    return texts


def _aligned(photo_ids: list[str], texts: dict[str, str]) -> list[str]:
    """Per-photo texts in story order, placeholders rewritten.

    All or nothing, because the schema wants ``[]`` or exactly one text per image.
    """
    out = [rewrite_placeholders(texts.get(pid, "")) for pid in photo_ids]
    return out if all(out) else []


def build_records(
    stories: list[RawStory],
    *,
    split: str,
    raw_image_dir: Path,
    out_image_dir: Path,
    max_image_side: int,
    dii: dict[str, str] | None = None,
    dis: dict[str, str] | None = None,
) -> tuple[list[StoryRecord], dict[str, int]]:
    """Steps 2–5: return ``(records, drop_counts)``; log how many stories were dropped and why.

    A story is dropped as a whole (``drop_counts`` says why) if it has a bad number of
    photos, no text, an empty sentence, or any photo that is missing or cannot be decoded.
    If a story has both missing and broken photos, it counts as ``missing_image``.
    Images are only resized for stories that are kept.
    """
    dii = dii or {}
    dis = dis or {}
    drops = dict.fromkeys(DROP_REASONS, 0)
    photo_status: dict[str, Path | str] = {}  # photo id -> valid raw path, or drop reason

    def check(photo_id: str) -> Path | str:
        if photo_id not in photo_status:
            raw = find_image(raw_image_dir, photo_id)
            if raw is None:
                photo_status[photo_id] = "missing_image"
            elif not is_valid_image(raw):
                photo_status[photo_id] = "broken_image"
            else:
                photo_status[photo_id] = raw
        return photo_status[photo_id]

    records: list[StoryRecord] = []
    for story in stories:
        if not MIN_IMAGES <= len(story.photo_ids) <= MAX_IMAGES:
            drops["bad_length"] += 1
            continue
        if not any(story.sentences):
            drops["empty_story"] += 1
            continue
        if not all(story.sentences):
            drops["empty_sentence"] += 1
            continue
        text = rewrite_placeholders(" ".join(story.sentences))
        checks = [check(pid) for pid in story.photo_ids]
        failed = {c for c in checks if isinstance(c, str)}
        if failed:
            # "missing_image" wins: it means the download is incomplete (blocks dev-200 lists).
            drops["missing_image" if "missing_image" in failed else "broken_image"] += 1
            continue

        images = []
        for pid, raw in zip(story.photo_ids, checks, strict=True):
            dst = out_image_dir / f"{pid}.jpg"
            resize_long_side(raw, dst, max_side=max_image_side)
            images.append(dst.as_posix())
        records.append(
            StoryRecord(
                id=story.story_id,
                dataset="vist",
                split=split,
                images=images,
                story=text,
                dis=_aligned(story.photo_ids, dis),
                dii=_aligned(story.photo_ids, dii),
            )
        )

    log.info(
        "VIST %s: kept %d of %d stories (%d with DII), dropped %s",
        split,
        len(records),
        len(stories),
        sum(bool(r.dii) for r in records),
        drops,
    )
    return records, drops


def _ensure_dev_ids(
    cfg: ExperimentConfig, split: str, records: list[StoryRecord], drops: dict[str, int]
) -> None:
    """Write the committed dev-subset id list once; never overwrite an existing one.

    Raises if an existing list names stories that were not built here, because eval
    would then silently run on fewer stories than the committed subset.
    """
    size = cfg.data.dev_subset_size
    path = subset_path(cfg.data.splits_dir, "vist", split, size)
    ids = {r.id for r in records}
    if path.exists():
        absent = [i for i in read_ids(path) if i not in ids]
        if absent:
            raise RuntimeError(
                f"{path} lists {len(absent)} ids that were not built here (e.g. {absent[0]}); "
                f"check the {split} images (drop counts: {drops})"
            )
        return
    if not ids:
        log.warning("not writing %s: no stories were built", path)
        return
    if drops["missing_image"]:
        log.warning(
            "not writing %s: %d stories lack images. Download all %s images first.",
            path,
            drops["missing_image"],
            split,
        )
        return
    write_ids(path, sample_ids(ids, size, cfg.experiment.seed))
    log.info("wrote %s", path)


def prepare(cfg: ExperimentConfig, split: str) -> Path:
    """Run steps 1–6 for one split; return the written JSONL path.

    Fails (and writes nothing) if the split's raw image folder is missing or no story
    could be built, so a bad run never replaces a good JSONL with an empty one.
    """
    raw_dir = Path(cfg.data.raw_dir) / "vist"
    sis_path = raw_dir / "sis" / f"{split}.story-in-sequence.json"
    if not sis_path.is_file():
        raise FileNotFoundError(f"{sis_path} not found; see data/README.md to download VIST")
    raw_image_dir = raw_dir / "images" / split
    if not raw_image_dir.is_dir():
        raise FileNotFoundError(
            f"{raw_image_dir} not found; download the VIST {split} images first "
            "(or run only the splits you have, e.g. --split val)"
        )
    with sis_path.open(encoding="utf-8") as f:
        stories = group_sis(json.load(f))

    records, drops = build_records(
        stories,
        split=split,
        raw_image_dir=raw_image_dir,
        out_image_dir=Path(cfg.data.image_dir) / "vist" / split,
        max_image_side=cfg.data.max_image_side,
        dii=per_photo_text(raw_dir / "dii" / f"{split}.description-in-isolation.json"),
        dis=per_photo_text(raw_dir / "dis" / f"{split}.description-in-sequence.json"),
    )
    out_path = Path(cfg.data.jsonl_dir) / f"vist_{split}.jsonl"
    if not records:
        raise RuntimeError(
            f"VIST {split}: no stories were built (dropped {drops}); "
            f"is {raw_image_dir} complete? {out_path} was not changed."
        )
    n = write_jsonl(out_path, records)
    log.info("VIST %s: wrote %d records to %s", split, n, out_path)

    if split in ("val", "test"):
        _ensure_dev_ids(cfg, split, records, drops)
    return out_path
