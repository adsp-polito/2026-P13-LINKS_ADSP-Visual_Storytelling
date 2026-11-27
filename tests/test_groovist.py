"""GROOViST maths and wiring, with a fake backend: no models, images or downloads."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from p13.common.config import ConfigError, load_config
from p13.common.jsonl import write_jsonl
from p13.common.schema import GeneratedStory, StoryRecord
from p13.eval import groovist as gv
from p13.eval.metrics import evaluate

THETA = 0.6


def _tok(text, pos="NOUN"):
    words = text.split()
    return [(w, w.rstrip("s"), pos, i == len(words) - 1) for i, w in enumerate(words)]


class FakeBackend:
    def __init__(self, chunks, texts, regions):
        self.chunks, self.texts, self.regions = chunks, texts, regions
        self.region_calls = []
        self.text_calls = 0

    def noun_chunks(self, text):
        return self.chunks[text]

    def embed_texts(self, texts):
        self.text_calls += 1
        return np.array([self.texts[t] for t in texts], dtype=float)

    def embed_regions(self, image_path):
        self.region_calls.append(image_path)
        return np.array(self.regions[image_path], dtype=float)


def _cos_vec(cos):
    return [cos, math.sqrt(1 - cos**2)]


def test_clean_text_replaces_vist_placeholders():
    assert gv.clean_text("[male] met [female] in [location].") == "male met female in location."


def test_select_phrases_drops_single_pronouns_and_repeats():
    chunks = [
        _tok("the dog"),
        [("they", "they", "PRON", True)],
        [("his", "his", "PRON", False), ("ball", "ball", "NOUN", True)],
        _tok("the dog"),
        [],
    ]
    kept = [" ".join(t[0] for t in c) for c in gv.select_phrases(chunks)]
    assert kept == ["the dog", "his ball"]


def test_concreteness_is_the_head_noun_with_lemma_then_default_fallback():
    lexicon = {"the": 1.5, "his": 3.1, "big": 3.7, "dog": 4.5}
    the_big_dogs = [
        ("the", "the", "DET", False),
        ("big", "big", "ADJ", False),
        ("dogs", "dog", "NOUN", True),
    ]
    assert gv.concreteness(the_big_dogs, lexicon) == 4.5
    assert gv.concreteness([("dog", "dog", "NOUN", True)], lexicon) == 4.5
    assert gv.concreteness(
        [("his", "his", "PRON", False), ("zzz", "zzz", "NOUN", True)], lexicon
    ) == (gv.DEFAULT_WEIGHT)
    assert (
        gv.concreteness([("the", "the", "DET", False), ("dog", "dog", "NOUN", False)], lexicon)
        == 4.5
    )


def test_penalty_keeps_grounded_and_shifts_ungrounded():
    assert gv.penalise(0.8, THETA) == 0.8
    assert gv.penalise(THETA, THETA) == THETA
    assert gv.penalise(0.5, THETA) == pytest.approx(0.5 - THETA)


def test_story_score_is_tanh_of_weighted_mean():
    phrases = [gv.Phrase("a", 2.0, 0.8), gv.Phrase("b", 4.0, 0.5)]
    expected = math.tanh((2.0 * 0.8 + 4.0 * (0.5 - THETA)) / 2)
    assert gv.story_score(phrases, THETA) == pytest.approx(expected)
    assert -1 < gv.story_score(phrases, THETA) < 1


def test_raw_score_is_the_mean_before_tanh_and_grounded_share_counts_theta():
    phrases = [gv.Phrase("a", 2.0, 0.8), gv.Phrase("b", 4.0, 0.5), gv.Phrase("c", 1.0, THETA)]
    raw = (2.0 * 0.8 + 4.0 * (0.5 - THETA) + 1.0 * THETA) / 3
    assert gv.raw_score(phrases, THETA) == pytest.approx(raw)
    assert gv.story_score(phrases, THETA) == pytest.approx(math.tanh(raw))
    assert gv.grounded_share(phrases, THETA) == pytest.approx(2 / 3)
    assert math.isnan(gv.raw_score([], THETA)) and math.isnan(gv.grounded_share([], THETA))


def test_story_without_phrases_gets_no_score():
    assert math.isnan(gv.story_score([], THETA))


def test_alignment_is_scaled_max_over_all_regions_of_all_images():
    phrases = np.array([[1.0, 0.0], [0.0, 2.0]])
    image_a = np.array([_cos_vec(0.2), [-1.0, 0.0]])
    image_b = np.array([_cos_vec(0.9)])
    got = gv.max_alignments(phrases, [image_a, image_b])
    assert got == pytest.approx([2.5 * 0.9, 2.5 * math.sqrt(1 - 0.2**2)])
    assert gv.max_alignments(phrases, [image_b, image_a]) == pytest.approx(got)


def test_select_boxes_scales_to_padded_square_and_clips():
    boxes = np.array([[0.5, 0.25, 0.5, 0.5]])
    got = gv.select_boxes(boxes, np.array([1.0]), width=200, height=100, k=5)
    assert got == [(50, 0, 150, 100)]


def test_select_boxes_keeps_top_k_after_nms_and_skips_empty_boxes():
    boxes = np.array(
        [
            [0.25, 0.25, 0.2, 0.2],
            [0.26, 0.25, 0.2, 0.2],
            [0.75, 0.75, 0.2, 0.2],
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.25, 0.2, 0.2],
        ]
    )
    objectness = np.array([0.9, 0.8, 0.7, 0.95, 0.1])
    got = gv.select_boxes(boxes, objectness, width=100, height=100, k=2)
    assert got == [(15, 15, 35, 35), (65, 65, 85, 85)]
    assert len(gv.select_boxes(boxes, objectness, width=100, height=100, k=10)) == 3


def test_load_lexicon_reads_word_and_mean(tmp_path):
    path = tmp_path / "c.tsv"
    path.write_text("Word\tConc.M\nDog\t4.85\nidea\t1.61\n", encoding="utf-8")
    assert gv.load_lexicon(str(path)) == {"dog": 4.85, "idea": 1.61}
    with pytest.raises(FileNotFoundError, match="download_concreteness"):
        gv.load_lexicon(str(tmp_path / "missing.tsv"))


def _setup(
    tmp_path,
    story_dict,
    generated_dict,
    monkeypatch,
    stories,
    threshold="{vist: 0.6}",
    metric_names="groovist",
):
    write_jsonl(
        tmp_path / "processed" / "vist_val.jsonl",
        [StoryRecord.model_validate({**story_dict, "id": i}) for i in stories],
    )
    gens = [
        GeneratedStory.model_validate({**generated_dict, "id": i, "story": text})
        for i, text in stories.items()
    ]
    write_jsonl(tmp_path / "gens.jsonl", gens)
    images = story_dict["images"]
    backend = FakeBackend(
        chunks={"A dog.": [_tok("a dog")], "It ran.": [[("it", "it", "PRON", True)]]},
        texts={"a dog": [1.0, 0.0]},
        regions={images[0]: [_cos_vec(0.3)], images[1]: [_cos_vec(0.32)], images[2]: [[0, 1]]},
    )
    monkeypatch.setattr(gv, "load_backend", lambda cfg: backend)
    monkeypatch.setattr(gv, "load_lexicon", lambda path: {"a": 1.0, "dog": 5.0})
    cfg = load_config(
        "configs/e0_zeroshot.yaml",
        [
            f"data.jsonl_dir={tmp_path / 'processed'}",
            f"output.dir={tmp_path / 'outputs'}",
            f"eval.metrics=[{metric_names}]",
            f"eval.groovist_threshold={threshold}",
        ],
    )
    return cfg, tmp_path / "gens.jsonl", backend


def test_evaluate_scores_groovist_and_leaves_no_phrase_stories_empty(
    tmp_path, story_dict, generated_dict, monkeypatch
):
    cfg, gens, backend = _setup(
        tmp_path, story_dict, generated_dict, monkeypatch, {"s1": "A dog.", "s2": "It ran."}
    )
    with evaluate(cfg, [gens]).open(encoding="utf-8", newline="") as f:
        rows = {r["id"]: r["groovist"] for r in csv.DictReader(f)}
    assert float(rows["s1"]) == pytest.approx(math.tanh(5.0 * 2.5 * 0.32))
    assert rows["s2"] == ""
    assert sorted(set(backend.region_calls)) == sorted(story_dict["images"])


def test_groovist_needs_a_threshold_for_every_dataset(
    tmp_path, story_dict, generated_dict, monkeypatch
):
    cfg, gens, _ = _setup(
        tmp_path, story_dict, generated_dict, monkeypatch, {"s1": "A dog."}, threshold="{vwp: 0.6}"
    )
    with pytest.raises(ValueError, match=r"groovist_threshold for \['vist'\]"):
        evaluate(cfg, [gens])


def test_threshold_config_is_a_mapping_of_known_datasets():
    cfg = load_config("configs/e0_zeroshot.yaml", ["eval.groovist_threshold={vist: 0.61, vwp: 1}"])
    assert cfg.eval.groovist_threshold == {"vist": 0.61, "vwp": 1.0}
    for bad in ["{coco: 0.6}", "0.6", "{vist: high}"]:
        with pytest.raises(ConfigError):
            load_config("configs/e0_zeroshot.yaml", [f"eval.groovist_threshold={bad}"])


def _phrase_rows(cfg):
    path = cfg.run_dir / gv.PHRASES_FILE
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_evaluate_writes_one_row_per_phrase_and_replaces_it_each_run(
    tmp_path, story_dict, generated_dict, monkeypatch
):
    cfg, gens, _ = _setup(
        tmp_path, story_dict, generated_dict, monkeypatch, {"s1": "A dog.", "s2": "It ran."}
    )
    evaluate(cfg, [gens])
    evaluate(cfg, [gens])
    rows = _phrase_rows(cfg)
    assert len(rows) == 1
    row = rows[0]
    assert (row["id"], row["phrase"], row["grounded"]) == ("s1", "a dog", True)
    assert row["model"] == generated_dict["model"] and row["adapter"] is None
    assert row["weight"] == 5.0
    assert row["alignment"] == pytest.approx(0.8)
    assert row["value"] == pytest.approx(4.0)


def test_evaluate_removes_old_phrase_file_when_groovist_is_not_asked(
    tmp_path, story_dict, generated_dict, monkeypatch
):
    cfg, gens, _ = _setup(tmp_path, story_dict, generated_dict, monkeypatch, {"s1": "A dog."})
    evaluate(cfg, [gens])
    assert (cfg.run_dir / gv.PHRASES_FILE).exists()
    cfg.eval.metrics = ["length"]
    evaluate(cfg, [gens])
    assert not (cfg.run_dir / gv.PHRASES_FILE).exists()


def test_phrase_file_name_is_a_run_side_file():
    from p13.eval import metrics

    assert gv.PHRASES_FILE in metrics.RUN_SIDE_FILES


def _ref(story_dict, sid, story, images):
    return StoryRecord.model_validate({**story_dict, "id": sid, "story": story, "images": images})


def _calibration_setup(tmp_path, story_dict, monkeypatch, overrides=()):
    imgs = {k: [f"{k}{n}.jpg" for n in range(3)] for k in "abc"}
    refs = [
        _ref(story_dict, "r1", "A dog.", imgs["a"]),
        _ref(story_dict, "r2", "A cat.", imgs["b"]),
        _ref(story_dict, "r3", "It ran.", imgs["c"]),
    ]
    write_jsonl(tmp_path / "processed" / "vist_val.jsonl", refs)
    backend = FakeBackend(
        chunks={"A dog.": [_tok("a dog")], "A cat.": [_tok("a cat")], "It ran.": []},
        texts={"a dog": [1.0, 0.0], "a cat": [0.0, 1.0]},
        regions={
            **{p: [[1.0, 0.0]] for p in imgs["a"]},
            **{p: [[0.0, 1.0]] for p in imgs["b"]},
            **{p: [[0.6, 0.8]] for p in imgs["c"]},
        },
    )
    monkeypatch.setattr(gv, "load_backend", lambda cfg: backend)
    monkeypatch.setattr(gv, "load_lexicon", lambda path: {})
    cfg = load_config(
        "configs/e0_zeroshot.yaml",
        [
            f"data.jsonl_dir={tmp_path / 'processed'}",
            f"output.dir={tmp_path / 'outputs'}",
            "data.datasets=[vist]",
            "data.subset=full",
            *overrides,
        ],
    )
    return cfg, refs


def test_calibrate_sets_theta_to_mean_reference_alignment_and_checks_mismatch(
    tmp_path, story_dict, monkeypatch
):
    cfg, _ = _calibration_setup(tmp_path, story_dict, monkeypatch)
    report = json.loads(gv.calibrate(cfg).read_text(encoding="utf-8"))["vist"]
    assert report["theta"] == pytest.approx(2.5)
    assert (report["n_stories"], report["n_phrases"], report["n_compared"]) == (3, 2, 2)
    assert report["own_mean_score"] == pytest.approx(math.tanh(2.5))
    assert report["other_mean_score"] == pytest.approx((math.tanh(-2.5) + math.tanh(-0.5)) / 2)
    assert report["own_wins_share"] == 1.0


def test_calibrate_uses_the_dev_subset_ids(tmp_path, story_dict, monkeypatch):
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "vist_val_dev200.txt").write_text("r1\nr3\n", encoding="utf-8")
    cfg, _ = _calibration_setup(
        tmp_path,
        story_dict,
        monkeypatch,
        ["data.subset=dev200", f"data.splits_dir={tmp_path / 'splits'}"],
    )
    report = json.loads(gv.calibrate(cfg).read_text(encoding="utf-8"))["vist"]
    assert (report["n_stories"], report["n_phrases"]) == (2, 1)
    assert report["theta"] == pytest.approx(2.5)


def test_calibrate_refuses_test_split_and_missing_references(tmp_path, story_dict, monkeypatch):
    cfg, _ = _calibration_setup(tmp_path, story_dict, monkeypatch, ["data.split=test"])
    with pytest.raises(ValueError, match="never on the test split"):
        gv.calibrate(cfg)
    cfg, _ = _calibration_setup(tmp_path, story_dict, monkeypatch, ["data.datasets=[vwp]"])
    with pytest.raises(FileNotFoundError, match=r"vwp_val\.jsonl"):
        gv.calibrate(cfg)


def test_mismatch_partner_shares_no_image(story_dict):
    refs = [
        _ref(story_dict, "r1", "x", ["p1", "p2", "p3"]),
        _ref(story_dict, "r2", "x", ["p3", "p4", "p5"]),
        _ref(story_dict, "r3", "x", ["p6", "p7", "p8"]),
    ]
    assert [gv._partner(refs, i).id for i in range(3)] == ["r3", "r3", "r1"]
    assert gv._partner(refs[:2], 0) is None


def test_cli_calibrate_flag(tmp_path, monkeypatch):
    from p13 import cli

    calls = []
    monkeypatch.setattr(gv, "calibrate", lambda cfg: calls.append(cfg.data.split))
    base = ["--config", "configs/e0_zeroshot.yaml", "--set", f"output.dir={tmp_path}"]
    assert cli.eval_main([*base, "--calibrate-groovist"]) == 0
    assert calls == ["val"]
    for extra in (["--with-human"], ["--stories", "x.jsonl"], []):
        with pytest.raises(SystemExit):
            cli.eval_main([*base, "--calibrate-groovist", *extra] if extra else base)


def test_evaluate_writes_all_groovist_columns_from_one_model_pass(
    tmp_path, story_dict, generated_dict, monkeypatch
):
    cfg, gens, backend = _setup(
        tmp_path,
        story_dict,
        generated_dict,
        monkeypatch,
        {"s1": "A dog.", "s2": "It ran."},
        metric_names="groovist,groovist_raw,groovist_grounded_share",
    )
    with evaluate(cfg, [gens]).open(encoding="utf-8", newline="") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    assert float(rows["s1"]["groovist_raw"]) == pytest.approx(5.0 * 0.8)
    assert float(rows["s1"]["groovist"]) == pytest.approx(math.tanh(4.0))
    assert float(rows["s1"]["groovist_grounded_share"]) == 1.0
    assert rows["s2"]["groovist_raw"] == rows["s2"]["groovist_grounded_share"] == ""
    assert backend.text_calls == 1
    assert len(_phrase_rows(cfg)) == 1
