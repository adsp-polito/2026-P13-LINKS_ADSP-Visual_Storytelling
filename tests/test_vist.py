"""VIST tests: placeholder helper (Nioosha) and loader functions (Parsa)."""

import json

from p13.data.vist import RawStory, group_sis, per_photo_text, rewrite_placeholders


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
    flat = _sis("1", "p2", 1, "bye .")[0], _sis("1", "p1", 0, "hi .")[0]
    stories = group_sis({"annotations": list(flat)})
    assert stories == [RawStory("1", ["p1", "p2"], ["hi .", "bye ."])]


def test_per_photo_text_missing_file_returns_empty(tmp_path):
    assert per_photo_text(tmp_path / "missing.json") == {}


def test_per_photo_text_keeps_first_description_per_photo(tmp_path):
    path = tmp_path / "val.description-in-isolation.json"
    anns = [_dii("p1", "a red car ."), _dii("p1", "another car ."), _dii("p2", "a dog .")]
    path.write_text(json.dumps({"annotations": anns}), encoding="utf-8")
    assert per_photo_text(path) == {"p1": "a red car .", "p2": "a dog ."}
