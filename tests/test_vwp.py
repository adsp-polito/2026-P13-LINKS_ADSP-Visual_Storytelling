"""Tests for the VWP loader. Tiny fake CSVs and images only (CI has no datasets)."""

import csv
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from p13.common.config import parse_config_dict
from p13.common.jsonl import read_jsonl
from p13.common.schema import StoryRecord
from p13.common.subset import read_ids
from p13.data.vwp import (
    DROP_REASONS,
    RawStory,
    anonymiser_placeholder,
    build_records,
    image_key,
    image_urls,
    mask_leaked_names,
    prepare,
    read_csv,
    rewrite_placeholders,
    shots_of,
    story_text,
)

_NO_DROPS = dict.fromkeys(DROP_REASONS, 0)

# ---------------------------------------------------------------- text


def test_single_character_uses_a_then_the():
    assert rewrite_placeholders("[male0] ran . [male0] fell .") == "a man ran . the man fell ."


def test_several_characters_of_a_kind_are_numbered():
    text = "[male0] met [male1] . [male2] came . [male1] left with [male0] ."
    assert rewrite_placeholders(text) == (
        "a man met another man . a third man came . the second man left with the first man ."
    )


def test_kinds_are_counted_separately():
    text = "[male0] and [female0] saw [location] . [female0] smiled at [male0] ."
    assert rewrite_placeholders(text) == (
        "a man and a woman saw a place . the woman smiled at the man ."
    )


def test_organization_gets_an_and_case_is_ignored():
    assert rewrite_placeholders("[ORGANIZATION0] called .") == "an organization called ."


def test_leaked_name_is_masked():
    assert story_text("carey waves at [male0] .", ["Carey"], "Carey waves at Tom.") == (
        "a person waves at a man."
    )


def test_two_leaked_names_and_longer_name_first():
    text = "al and alan talk . al leaves ."
    original = "Al and Alan talk. Al leaves."
    assert story_text(text, ["Al", "Alan"], original) == (
        "a person and another person talk. the first person leaves."
    )


def test_name_typed_in_lower_case_is_still_masked():
    assert story_text("harrison waved .", ["Harrison"], "harrison waved.") == "a person waved."


def test_name_with_dots_is_masked():
    assert story_text("t.j. 's dog ran . j.a judged .", ["T.J.", "J.A."], "x") == (
        "a person's dog ran. another person judged."
    )


def test_single_letter_name_does_not_eat_words():
    assert mask_leaked_names("w. went with will .", ["W."], "W. went with Will.") == (
        "[person0] went with will ."
    )


def test_word_name_is_kept_when_story_uses_the_word():
    # "Guy" is a character, but "a guy" is also a normal word here: masking would break it.
    text = "a guy saw guy ."
    assert mask_leaked_names(text, ["Guy"], "A guy saw Guy.") == text


def test_word_name_is_masked_when_story_never_uses_the_word():
    assert story_text("art is hacking .", ["Art"], "Art is hacking.") == "a person is hacking."


def test_names_not_in_the_text_change_nothing():
    assert mask_leaked_names("[male0] ran .", ["Tom", "Guy"], "Tom ran.") == "[male0] ran ."


def test_name_that_is_only_dots_is_ignored():
    # "." would otherwise match every word boundary and fill the text with placeholders.
    assert story_text("tom ran .", [".", "Tom"], "Tom ran.") == "a person ran."


def test_missed_name_reuses_the_anonymisers_placeholder():
    # The anonymiser replaced the first two "John"s but missed the third one: keep one person.
    original = "John met Ann. Later John's wife and John himself left."
    anonymised = "[male0] met [female0] . later [male0] 's wife and john himself left ."
    assert story_text(anonymised, ["John", "Ann"], original) == (
        "a man met a woman. later the man's wife and the man himself left."
    )


def test_conflicting_placeholders_fall_back_to_a_person():
    original = "John met Bob. John left. John slept."
    anonymised = "[male0] met [male1] . [male1] left . john slept ."
    assert story_text(anonymised, ["John"], original) == (
        "a man met another man. the second man left. a person slept."
    )


def test_placeholder_is_found_right_after_a_full_stop():
    # "around." is one word in the original but "around" "." in the anonymised text.
    original = "They looked around. John ran. Later John slept."
    anonymised = "they looked around . [male0] ran . later john slept ."
    assert anonymiser_placeholder("John", original, anonymised) == "[male0]"


@pytest.mark.parametrize(
    ("original", "anonymised"),
    [
        ("John hasn't left. Later John slept.", "[male0] has n't left . later john slept ."),
        (
            "John didn\u2019t stop. Later John slept.",
            "[male0] did n\u2019t stop . later john slept .",
        ),
    ],
)
def test_placeholder_is_found_next_to_nt(original, anonymised):
    # "hasn't" must split like the anonymiser's "has n't" (straight or curly apostrophe).
    assert anonymiser_placeholder("John", original, anonymised) == "[male0]"


def test_titles_and_initials_keep_their_dot_when_lining_up():
    original = "Dr. Lee met T.J. at home. Later T.J. slept."
    anonymised = "dr. [male0] met [male1] at home . later t.j. slept ."
    assert anonymiser_placeholder("T.J.", original, anonymised) == "[male1]"


def test_dotted_name_keeps_the_full_stop_at_a_shot_boundary():
    # VWP joins the per-shot texts with two spaces; the first "T.J." also ends a sentence.
    original = "Keith tried to grab T.J. T.J. was able to get loose."
    anonymised = "[male0] tried to grab t.j.  t.j. was able to get loose ."
    assert story_text(anonymised, ["T.J."], original) == (
        "a man tried to grab a person.  the person was able to get loose."
    )


def test_dotted_name_mid_sentence_gets_no_full_stop():
    assert story_text("t.j. and [male0] drive .", ["T.J."], "T.J. and Kevin drive.") == (
        "a person and a man drive."
    )


def test_dotted_name_full_stop_found_in_the_original_story():
    # Same shot, one space: the capital "It" in the original shows the sentence ended.
    original = "Ann played with her brother W. It was fun. W. smiled."
    anonymised = "[female0] played with her brother w. it was fun . w. smiled ."
    assert story_text(anonymised, ["W."], original) == (
        "a woman played with her brother a person. it was fun. the person smiled."
    )


def test_dotted_name_never_gets_a_second_full_stop():
    assert story_text("she saw t.j. .", ["T.J."], "She saw T.J.") == "she saw a person."


def test_up_to_ten_characters_of_a_kind_get_words():
    text = " ".join(f"[male{i}]" for i in range(10)) + " . [male9] left ."
    out = rewrite_placeholders(text)
    assert "a ninth man" in out and "a tenth man" in out
    assert out.endswith("the tenth man left .") and "no." not in out


# ---------------------------------------------------------------- CSV

_URL = "https://example.org/VST/alpha/tt0000001_0001{tail}/shot_{n:04d}_img_0.jpg"


def _url(n: int, tail: str = "_0") -> str:
    return _URL.format(tail=tail, n=n)


def _sid(n: int, tail: str = "_0") -> str:
    """Image key of ``_url(n, tail)``: folder + shot file."""
    return f"tt0000001_0001{tail}_shot_{n:04d}_img_0"


def _row(
    story_id: str,
    scene: str,
    shots: list[int],
    anonymised: str = "[male0] smiled .",
    *,
    story: str = "Tom smiled.",
    chars: tuple[str, ...] = (),
    links: list[str] | None = None,
    n_listed: int | None = None,
    n_texts: int | None = None,
) -> dict[str, str]:
    links = links if links is not None else [_url(n) for n in shots]
    n_texts = len(links) if n_texts is None else n_texts
    ids = [_sid(n) for n in shots][: n_listed if n_listed is not None else len(shots)]
    row = {f"link{i}": (links[i] if i < len(links) else "") for i in range(10)}
    row |= {f"text{i}": ("part ." if i < n_texts else "") for i in range(10)}
    row |= {f"char{i}": (chars[i] if i < len(chars) else "{}") for i in range(5)}
    row |= {
        "story_id": story_id,
        "scene_full_id": scene,
        "img_id_list": str(ids),
        "story": story,
        "anonymised_story": anonymised,
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_image_key_accepts_both_folder_shapes():
    assert image_key(_url(667)) == "tt0000001_0001_0_shot_0667_img_0"
    assert image_key(_url(667, tail="")) == "tt0000001_0001_shot_0667_img_0"  # no last _<n>


def test_same_shot_in_two_folders_gets_two_keys():
    # VWP train has shots that sit in two scene folders; they must not share a file.
    assert image_key(_url(5, tail="_0")) != image_key(_url(5, tail="_1"))


def test_image_key_rejects_unexpected_url():
    with pytest.raises(ValueError, match="unexpected VWP image URL"):
        image_key("https://example.org/picture.jpg")


def test_read_csv_keeps_link_order_and_cleans_text(tmp_path):
    rows = [_row("7", "sc1", [9, 2, 5], "[male0] ran . tom fell .", chars=("Tom",))]
    (story,) = read_csv(_write_csv(tmp_path / "v.csv", rows))

    assert story.story_id == "7" and story.scene_id == "sc1"
    assert story.image_keys == (_sid(9), _sid(2), _sid(5))  # CSV order, not sorted
    assert (story.n_listed, story.n_texts) == (3, 3)
    assert story.text == "a man ran. a person fell."


def test_read_csv_treats_curly_braces_as_empty_link(tmp_path):
    rows = [_row("1", "sc", [1, 2, 3], links=[_url(1), _url(2), _url(3), "{}"], n_texts=4)]
    (story,) = read_csv(_write_csv(tmp_path / "v.csv", rows))

    assert len(story.urls) == 3 and story.n_texts == 4


def test_read_csv_marks_a_gap_in_the_links_as_not_aligned(tmp_path):
    # Same counts (3 links, 3 texts, 3 ids), but link2 is empty: shots would shift.
    links = [_url(1), _url(2), "", _url(3)]
    rows = [_row("1", "sc", [1, 2, 3], links=links, n_texts=3)]
    (story,) = read_csv(_write_csv(tmp_path / "v.csv", rows))

    assert (len(story.urls), story.n_listed, story.n_texts) == (3, 3, 3)
    assert story.aligned is False


def test_read_csv_rejects_duplicate_story_ids(tmp_path):
    rows = [_row("5", "sc", [1, 2, 3]), _row("5", "sc", [4, 5, 6])]

    with pytest.raises(ValueError, match="duplicate story_id"):
        read_csv(_write_csv(tmp_path / "v.csv", rows))


def test_read_csv_reads_shots_from_img_id_list(tmp_path):
    row = _row("1", "sc", [1, 2, 3])
    row["img_id_list"] = str(["tt0047396_0667_0", "tt0047396_0668_1", "not_a_shot"])
    (story,) = read_csv(_write_csv(tmp_path / "v.csv", [row]))
    assert story.listed_shots == {("tt0047396", "0667"), ("tt0047396", "0668")}
    assert story.n_listed == 3


def test_shots_of_ignores_the_scene_folder_and_adds_listed_shots():
    a = _story("a", [1, 2])
    # The same shot 2 in another scene folder ("_5"), plus one shot only in img_id_list.
    b = replace(a, image_keys=(_sid(2, "_5"),), listed_shots=frozenset({("tt9", "0007")}))
    assert shots_of(a) == {("tt0000001", "0001"), ("tt0000001", "0002")}
    assert shots_of(b) == {("tt0000001", "0002"), ("tt9", "0007")}


def test_image_urls_maps_each_shot_once(tmp_path):
    rows = [_row("1", "sc", [1, 2, 3]), _row("2", "sc", [2, 3, 4])]
    urls = image_urls(read_csv(_write_csv(tmp_path / "v.csv", rows)))

    assert urls == {_sid(n): _url(n) for n in (1, 2, 3, 4)}


# ---------------------------------------------------------------- build_records


def _make_image(directory: Path, sid: str, size=(800, 400)) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "red").save(directory / f"{sid}.jpg")


def _story(story_id: str, shots: list[int], text: str = "a man smiled .", **kw) -> RawStory:
    sids = tuple(_sid(n) for n in shots)
    return RawStory(
        story_id=story_id,
        scene_id=kw.get("scene", "sc"),
        image_keys=sids,
        urls=tuple(_url(n) for n in shots),
        n_listed=kw.get("n_listed", len(sids)),
        n_texts=kw.get("n_texts", len(sids)),
        text=text,
        aligned=kw.get("aligned", True),
    )


def _build(stories, raw, out):
    return build_records(
        stories, split="val", raw_image_dir=raw, out_image_dir=out, max_image_side=448
    )


def test_build_records_keeps_shot_order_and_resizes(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    for n in (9, 2, 5):
        _make_image(raw, _sid(n))

    records, drops = _build([_story("s1", [9, 2, 5])], raw, out)

    assert drops == _NO_DROPS
    (rec,) = records
    assert (rec.id, rec.dataset, rec.split) == ("s1", "vwp", "val")
    assert rec.story == "a man smiled ."
    assert (rec.dis, rec.dii) == ([], [])
    assert [Path(p).name for p in rec.images] == [f"{_sid(n)}.jpg" for n in (9, 2, 5)]
    with Image.open(rec.images[0]) as im:
        assert im.size == (448, 224)


def test_build_records_drop_reasons(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    for n in range(1, 13):
        _make_image(raw, _sid(n))
    (raw / f"{_sid(12)}.jpg").write_bytes(b"not an image")
    stories = [
        _story("ok", [1, 2, 3]),
        _story("lost_link", [1, 2, 3], n_listed=4),  # CSV lost the 4th link
        _story("text_without_image", [1, 2, 3], n_texts=4),  # "{}" link case
        _story("dup", [1, 2, 3]),  # same shots and text as "ok"
        _story("too_short", [1, 2]),
        _story("empty", [4, 5, 6], text=""),
        _story("missing", [7, 8, 99]),
        _story("broken", [10, 11, 12]),
    ]

    records, drops = _build(stories, raw, out)

    assert [r.id for r in records] == ["ok"]
    assert drops == {
        "val_overlap": 0,
        "link_count_mismatch": 2,
        "duplicate": 1,
        "bad_length": 1,
        "empty_story": 1,
        "missing_image": 1,
        "broken_image": 1,
    }


def test_build_records_same_shots_different_text_is_not_a_duplicate(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    for n in (1, 2, 3):
        _make_image(raw, _sid(n))
    stories = [_story("a", [1, 2, 3], "a man ran ."), _story("b", [1, 2, 3], "a woman sat .")]

    records, drops = _build(stories, raw, out)

    assert [r.id for r in records] == ["a", "b"] and drops == _NO_DROPS


def test_build_records_counts_missing_before_broken(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    _make_image(raw, _sid(1))
    (raw / f"{_sid(2)}.jpg").write_bytes(b"broken")

    _, drops = _build([_story("s", [1, 2, 3])], raw, out)  # shot 3 is missing

    assert drops["missing_image"] == 1 and drops["broken_image"] == 0


def test_build_records_drops_a_story_with_a_link_gap(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    for n in (1, 2, 3):
        _make_image(raw, _sid(n))

    records, drops = _build([_story("gap", [1, 2, 3], aligned=False)], raw, out)

    assert records == [] and drops["link_count_mismatch"] == 1


def test_build_records_drops_val_overlap_first(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    for n in range(1, 7):
        _make_image(raw, _sid(n))
    stories = [
        _story("keep", [1, 2, 3]),
        _story("shares_shot_4", [4, 5, 6]),
        _story("gap_and_shares_shot_4", [4, 5, 6], "a woman sat .", aligned=False),
    ]

    records, drops = build_records(
        stories,
        split="train",
        raw_image_dir=raw,
        out_image_dir=out,
        max_image_side=448,
        exclude_shots={("tt0000001", "0004")},
    )

    assert [r.id for r in records] == ["keep"]
    assert drops == _NO_DROPS | {"val_overlap": 2}  # counted before the link gap


# ---------------------------------------------------------------- prepare

# story id -> (scene, shots); scenes A and B have two stories each.
_STORIES = {
    "0": ("A", [1, 2, 3]),
    "1": ("A", [1, 2, 4]),
    "2": ("B", [5, 6, 7]),
    "3": ("B", [5, 6, 8]),
}


def _write_split(root: Path, split: str) -> None:
    vwp = root / "raw" / "vwp"
    rows = []
    for story_id, (scene, shots) in _STORIES.items():
        rows.append(_row(story_id, scene, shots))
        for n in shots:
            _make_image(vwp / "images" / split, _sid(n))
    _write_csv(vwp / f"vwp_v2.1_{split}.csv", rows)


def _write_val_csv(root: Path, shots: list[int]) -> None:
    """Only the val CSV (no images): what prepare("train") reads to find the val shots."""
    _write_csv(root / "raw" / "vwp" / "vwp_v2.1_val.csv", [_row("v1", "V", shots)])


def _cfg(root: Path):
    return parse_config_dict(
        {
            "data": {
                "datasets": ["vwp"],
                "raw_dir": str(root / "raw"),
                "image_dir": str(root / "images"),
                "jsonl_dir": str(root / "processed"),
                "splits_dir": str(root / "splits"),
                "dev_subset_size": 2,
            }
        }
    )


def test_prepare_writes_jsonl_and_one_dev_id_per_scene(tmp_path):
    _write_split(tmp_path, "val")
    cfg = _cfg(tmp_path)

    out = prepare(cfg, "val")

    assert out == tmp_path / "processed" / "vwp_val.jsonl"
    records = read_jsonl(out, StoryRecord)
    assert [r.id for r in records] == ["0", "1", "2", "3"]  # the full split, not only dev ids
    assert records[0].story == "a man smiled."

    dev = read_ids(tmp_path / "splits" / "vwp_val_dev2.txt")
    scenes = {_STORIES[i][0] for i in dev}
    assert len(dev) == 2 and scenes == {"A", "B"}  # one story from each scene

    first = (tmp_path / "splits" / "vwp_val_dev2.txt").read_text(encoding="utf-8")
    prepare(cfg, "val")  # a second run must not change the committed id list
    assert (tmp_path / "splits" / "vwp_val_dev2.txt").read_text(encoding="utf-8") == first


def test_prepare_train_writes_no_dev_ids(tmp_path):
    _write_split(tmp_path, "train")
    _write_val_csv(tmp_path, [50, 51, 52])

    prepare(_cfg(tmp_path), "train")

    assert not (tmp_path / "splits").exists()


def test_prepare_train_drops_stories_that_share_a_shot_with_val(tmp_path):
    _write_split(tmp_path, "train")
    _write_val_csv(tmp_path, [3, 50, 51])  # shot 3 is only in train story "0"

    out = prepare(_cfg(tmp_path), "train")

    assert [r.id for r in read_jsonl(out, StoryRecord)] == ["1", "2", "3"]


def test_prepare_train_fails_without_val_csv(tmp_path):
    _write_split(tmp_path, "train")
    with pytest.raises(FileNotFoundError, match=r"vwp_v2\.1_val\.csv"):
        prepare(_cfg(tmp_path), "train")


def test_prepare_skips_dev_ids_when_images_are_missing(tmp_path):
    _write_split(tmp_path, "val")
    (tmp_path / "raw" / "vwp" / "images" / "val" / f"{_sid(8)}.jpg").unlink()

    prepare(_cfg(tmp_path), "val")

    assert not (tmp_path / "splits" / "vwp_val_dev2.txt").exists()


def test_prepare_fails_if_committed_dev_ids_are_not_built(tmp_path):
    _write_split(tmp_path, "val")
    dev = tmp_path / "splits" / "vwp_val_dev2.txt"
    dev.parent.mkdir(parents=True)
    dev.write_text("0\n99\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not built here"):
        prepare(_cfg(tmp_path), "val")


def test_prepare_keeps_old_jsonl_when_nothing_is_built(tmp_path):
    _write_split(tmp_path, "val")
    out = tmp_path / "processed" / "vwp_val.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text("old\n", encoding="utf-8")
    for img in (tmp_path / "raw" / "vwp" / "images" / "val").iterdir():
        img.unlink()

    with pytest.raises(RuntimeError, match="no stories were built"):
        prepare(_cfg(tmp_path), "val")
    assert out.read_text(encoding="utf-8") == "old\n"


def test_prepare_refuses_blind_test_split(tmp_path):
    with pytest.raises(ValueError, match="blind"):
        prepare(_cfg(tmp_path), "test")


def test_prepare_fails_without_csv(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"download_vwp\.py"):
        prepare(_cfg(tmp_path), "val")


def test_prepare_fails_without_image_folder(tmp_path):
    _write_split(tmp_path, "val")
    for img in (tmp_path / "raw" / "vwp" / "images" / "val").iterdir():
        img.unlink()
    (tmp_path / "raw" / "vwp" / "images" / "val").rmdir()

    with pytest.raises(FileNotFoundError, match="images"):
        prepare(_cfg(tmp_path), "val")
