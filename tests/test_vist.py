"""VIST tests: placeholder helper (Nioosha) and loader functions (Parsa)."""

import json
from pathlib import Path

from PIL import Image

from p13.common.config import parse_config_dict
from p13.common.jsonl import read_jsonl
from p13.common.schema import StoryRecord
from p13.common.subset import read_ids
from p13.data.vist import (
    RawStory,
    build_records,
    group_sis,
    per_photo_text,
    prepare,
    rewrite_placeholders,
)


def test_placeholders_rewritten():
    out = rewrite_placeholders("[male] and [female] went to [location] .")
    assert out == "a man and a woman went to a place ."
    assert "[" not in out


def test_repeated_placeholder_uses_definite_article():
    out = rewrite_placeholders("[male] arrived . [male] smiled at [organization] .")
    assert out == "a man arrived . the man smiled at an organization ."


def test_case_insensitive():
    assert rewrite_placeholders("[MALE] ran") == "a man ran"


def _sis(story_id, photo_id, order, text):
    """One SIS annotation, wrapped in a one-element list like the real VIST files."""
    return [
        {
            "story_id": story_id,
            "photo_flickr_id": photo_id,
            "worker_arranged_photo_order": order,
            "text": text,
        }
    ]


def _dii(photo_id, text):
    return [{"photo_flickr_id": photo_id, "photo_order_in_story": 0, "text": text}]


def test_group_sis_orders_photos_and_groups_stories():
    sis = {
        "annotations": [
            _sis("2", "b2", 1, "second ."),
            _sis("1", "a1", 0, "first ."),
            _sis("2", "b1", 0, "start ."),
            _sis("1", "a2", 1, "then ."),
        ]
    }
    assert group_sis(sis) == [
        RawStory("2", ["b1", "b2"], ["start .", "second ."]),
        RawStory("1", ["a1", "a2"], ["first .", "then ."]),
    ]


def test_group_sis_accepts_flat_annotations():
    flat = [_sis("1", "p2", 1, "bye .")[0], _sis("1", "p1", 0, "hi .")[0]]
    assert group_sis({"annotations": flat}) == [RawStory("1", ["p1", "p2"], ["hi .", "bye ."])]


def test_per_photo_text_missing_file_returns_empty(tmp_path):
    assert per_photo_text(tmp_path / "missing.json") == {}


def test_per_photo_text_keeps_first_description_per_photo(tmp_path):
    path = tmp_path / "val.description-in-isolation.json"
    anns = [_dii("p1", "a red car ."), _dii("p1", "another car ."), _dii("p2", "a dog .")]
    path.write_text(json.dumps({"annotations": anns}), encoding="utf-8")
    assert per_photo_text(path) == {"p1": "a red car .", "p2": "a dog ."}


# ---------------------------------------------------------------- build_records / prepare

_NO_DROPS = {"bad_length": 0, "empty_story": 0, "missing_image": 0, "broken_image": 0}
_STORIES = {
    "s1": ["s1_1", "s1_2", "s1_3"],
    "s2": ["s2_1", "s2_2", "s2_3"],
    "s3": ["s3_1", "s3_2", "s3_3"],
}


def _make_image(directory: Path, photo_id: str, size=(800, 400)) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "red").save(directory / f"{photo_id}.jpg")


def _story(story_id: str, photo_ids: list[str]) -> RawStory:
    return RawStory(story_id, photo_ids, ["[male] smiled ."] * len(photo_ids))


def test_build_records_resizes_and_rewrites(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    ids = ["p1", "p2", "p3"]
    for pid in ids:
        _make_image(raw, pid)
    dii = {pid: "[female] holds a cup ." for pid in ids}

    records, drops = build_records(
        [_story("s1", ids)],
        split="val",
        raw_image_dir=raw,
        out_image_dir=out,
        max_image_side=448,
        dii=dii,
    )

    assert drops == _NO_DROPS
    (rec,) = records
    assert (rec.id, rec.dataset, rec.split) == ("s1", "vist", "val")
    assert rec.story == "a man smiled . the man smiled . the man smiled ."
    assert rec.dii == ["a woman holds a cup ."] * 3
    assert rec.dis == []
    assert [Path(p).name for p in rec.images] == ["p1.jpg", "p2.jpg", "p3.jpg"]
    with Image.open(rec.images[0]) as im:
        assert im.size == (448, 224)


def test_build_records_drops_whole_sequence(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    for pid in ["a1", "a2", "a3", "b1", "b3", "c1", "c2", "c3"]:
        _make_image(raw, pid)
    (raw / "c2.jpg").write_bytes(b"not an image")  # broken file
    stories = [
        _story("A", ["a1", "a2", "a3"]),
        _story("B", ["b1", "b2", "b3"]),  # b2 is missing
        _story("C", ["c1", "c2", "c3"]),  # c2 cannot be decoded
        _story("D", ["d1", "d2"]),  # too short
    ]

    records, drops = build_records(
        stories, split="val", raw_image_dir=raw, out_image_dir=out, max_image_side=448
    )

    assert [r.id for r in records] == ["A"]
    assert drops == {**_NO_DROPS, "bad_length": 1, "missing_image": 1, "broken_image": 1}
    assert (out / "a1.jpg").exists()
    assert not (out / "b1.jpg").exists()  # nothing is resized for dropped stories
    assert not (out / "c1.jpg").exists()


def test_build_records_descriptions_are_all_or_nothing(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    ids = ["p1", "p2", "p3"]
    for pid in ids:
        _make_image(raw, pid)

    records, _ = build_records(
        [_story("s1", ids)],
        split="val",
        raw_image_dir=raw,
        out_image_dir=out,
        max_image_side=448,
        dii={"p1": "a cup .", "p2": "a dog ."},  # p3 has no description
    )

    assert records[0].dii == []


def _write_split(root: Path, split: str) -> None:
    """Tiny SIS + DII files and one image per photo under ``root/raw/vist``."""
    vist = root / "raw" / "vist"
    (vist / "sis").mkdir(parents=True)
    (vist / "dii").mkdir(parents=True)
    sis, dii = [], []
    for story_id, photo_ids in _STORIES.items():
        for order, pid in enumerate(photo_ids):
            sis.append(_sis(story_id, pid, order, "[male] smiled ."))
            dii.append(_dii(pid, "a photo ."))
            _make_image(vist / "images" / split, pid)
    sis_file = vist / "sis" / f"{split}.story-in-sequence.json"
    dii_file = vist / "dii" / f"{split}.description-in-isolation.json"
    sis_file.write_text(json.dumps({"annotations": sis}), encoding="utf-8")
    dii_file.write_text(json.dumps({"annotations": dii}), encoding="utf-8")


def _cfg(root: Path):
    return parse_config_dict(
        {
            "data": {
                "datasets": ["vist"],
                "raw_dir": str(root / "raw"),
                "image_dir": str(root / "images"),
                "jsonl_dir": str(root / "processed"),
                "splits_dir": str(root / "splits"),
                "dev_subset_size": 2,
            }
        }
    )


def test_prepare_writes_full_jsonl_and_dev_ids(tmp_path):
    _write_split(tmp_path, "val")
    cfg = _cfg(tmp_path)

    out = prepare(cfg, "val")

    assert out == tmp_path / "processed" / "vist_val.jsonl"
    records = read_jsonl(out, StoryRecord)
    assert [r.id for r in records] == ["s1", "s2", "s3"]  # the full split, not only dev ids
    assert records[0].story == "a man smiled . the man smiled . the man smiled ."
    assert records[0].dii == ["a photo ."] * 3

    dev = tmp_path / "splits" / "vist_val_dev2.txt"
    assert len(read_ids(dev)) == 2
    assert set(read_ids(dev)) <= {"s1", "s2", "s3"}

    first = dev.read_text(encoding="utf-8")
    prepare(cfg, "val")  # a second run must not change the committed id list
    assert dev.read_text(encoding="utf-8") == first


def test_prepare_skips_dev_ids_when_images_are_missing(tmp_path):
    _write_split(tmp_path, "val")
    (tmp_path / "raw" / "vist" / "images" / "val" / "s3_2.jpg").unlink()

    out = prepare(_cfg(tmp_path), "val")

    assert [r.id for r in read_jsonl(out, StoryRecord)] == ["s1", "s2"]
    assert not (tmp_path / "splits").exists()


def test_prepare_train_writes_no_dev_ids(tmp_path):
    _write_split(tmp_path, "train")

    prepare(_cfg(tmp_path), "train")

    assert not (tmp_path / "splits").exists()
