"""VIST → unified JSONL.  **Owner: Parsa.**"""

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
from p13.common.schema import StoryRecord
from p13.common.subset import read_ids, sample_ids, subset_path, write_ids
from p13.data.images import find_image, is_valid_image, resize_long_side

log = logging.getLogger(__name__)

PLACEHOLDERS: dict[str, str] = {
    "[male]": "a man",
    "[female]": "a woman",
    "[location]": "a place",
    "[organization]": "an organization",
}
_PLACEHOLDER_RE = re.compile(r"\[(male|female|location|organization)\]", re.IGNORECASE)
TEXT_FIELD = "text"


def rewrite_placeholders(text: str) -> str:
    """Replace VIST anonymisation tokens with generic noun phrases."""
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
    """Step 1: flat SIS annotation list → ordered stories."""
    by_story: dict[str, list[dict]] = defaultdict(list)
    for ann in _flatten(sis.get("annotations", [])):
        by_story[str(ann["story_id"])].append(ann)
    stories = []
    for story_id, anns in by_story.items():
        anns.sort(key=lambda a: int(a["worker_arranged_photo_order"]))
        stories.append(
            RawStory(
                story_id=story_id,
                photo_ids=[str(a["photo_flickr_id"]) for a in anns],
                sentences=[str(a[TEXT_FIELD]).strip() for a in anns],
            )
        )
    return stories


def per_photo_text(path: Path) -> dict[str, str]:
    """Step 5: ``photo_flickr_id`` → description from a DII or DIS file ({} if absent)."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    texts: dict[str, str] = {}
    for ann in _flatten(data.get("annotations", [])):
        texts.setdefault(
            str(ann["photo_flickr_id"]), rewrite_placeholders(str(ann[TEXT_FIELD]).strip())
        )
    return texts


def build_records(
    stories: list[RawStory],
    *,
    split: str,
    raw_image_dir: Path,
    out_image_dir: Path,
    max_image_side: int,
    dii_map: dict[str, str] | None = None,
) -> tuple[list[StoryRecord], dict[str, int]]:
    """Steps 2–5: return ``(records, drop_counts)``."""
    dii_map = dii_map or {}
    records: list[StoryRecord] = []
    drop_counts = {"len_out_of_bounds": 0, "missing_image": 0, "invalid_image": 0}

    for story in stories:
        if not (3 <= len(story.photo_ids) <= 10):
            drop_counts["len_out_of_bounds"] += 1
            continue

        valid = True
        image_paths = []
        for pid in story.photo_ids:
            raw_img = find_image(raw_image_dir, pid)
            if raw_img is None:
                drop_counts["missing_image"] += 1
                valid = False
                break
            if not is_valid_image(raw_img):
                drop_counts["invalid_image"] += 1
                valid = False
                break

            dst_img = out_image_dir / f"{pid}.jpg"
            resize_long_side(raw_img, dst_img, max_side=max_image_side)
            image_paths.append(str(dst_img))

        if not valid:
            continue

        full_story_text = rewrite_placeholders(" ".join(story.sentences))
        dii_list = [dii_map.get(pid, "") for pid in story.photo_ids] if dii_map else []
        if not any(dii_list):
            dii_list = []

        record = StoryRecord(
            id=story.story_id,
            dataset="vist",
            split=split,
            images=image_paths,
            story=full_story_text,
            dis=[],
            dii=dii_list,
        )
        records.append(record)

    log.info("VIST %s: built %d stories, dropped %s", split, len(records), drop_counts)
    return records, drop_counts


def prepare(cfg: ExperimentConfig, split: str) -> Path:
    """Run steps 1–6 for one split; return the written JSONL path."""
    raw_dir = Path(cfg.data.raw_dir) / "vist"
    sis_path = raw_dir / "sis" / f"{split}.story-in-sequence.json"
    dii_path = raw_dir / "dii" / f"{split}.description-in-isolation.json"
    raw_img_dir = raw_dir / "images" / split
    out_img_dir = Path(cfg.data.image_dir) / "vist" / split
    jsonl_dir = Path(cfg.data.jsonl_dir)

    with sis_path.open(encoding="utf-8") as f:
        sis_data = json.load(f)

    raw_stories = group_sis(sis_data)
    dii_map = per_photo_text(dii_path)

    records, _ = build_records(
        raw_stories,
        split=split,
        raw_image_dir=raw_img_dir,
        out_image_dir=out_img_dir,
        max_image_side=cfg.data.max_image_side,
        dii_map=dii_map,
    )

    if cfg.data.subset == "dev200" and split in ("val", "test"):
        s_path = subset_path(cfg.data.splits_dir, "vist", split, cfg.data.dev_subset_size)
        if not s_path.exists():
            all_ids = [r.id for r in records]
            sampled = sample_ids(all_ids, cfg.data.dev_subset_size, cfg.experiment.seed)
            write_ids(s_path, sampled)
        allowed_ids = set(read_ids(s_path))
        records = [r for r in records if r.id in allowed_ids]

    out_path = jsonl_dir / f"vist_{split}.jsonl"
    write_jsonl(out_path, records)
    return out_path
