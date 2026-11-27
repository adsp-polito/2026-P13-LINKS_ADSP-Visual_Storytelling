"""GROOViST maths and wiring, with a fake backend: no models, images or downloads."""

from __future__ import annotations

import csv
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
    return [(w, w.rstrip("s"), pos) for w in text.split()]


class FakeBackend:
    def __init__(self, chunks, texts, regions):
        self.chunks, self.texts, self.regions = chunks, texts, regions
        self.region_calls = []

    def noun_chunks(self, text):
        return self.chunks[text]

    def embed_texts(self, texts):
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
        [("they", "they", "PRON")],
        _tok("his", "PRON") + _tok("ball"),
        _tok("the dog"),
        [],
    ]
    kept = [" ".join(w for w, _, _ in c) for c in gv.select_phrases(chunks)]
    assert kept == ["the dog", "his ball"]


def test_concreteness_is_mean_of_known_words_with_lemma_fallback():
    lexicon = {"the": 1.5, "dog": 4.5}
    assert gv.concreteness([("the", "the", "DET"), ("dogs", "dog", "NOUN")], lexicon) == 3.0
    assert gv.concreteness([("the", "the", "DET"), ("zzz", "zzz", "X")], lexicon) == 1.5
    assert gv.concreteness([("zzz", "zzz", "X")], lexicon) == gv.DEFAULT_WEIGHT


def test_penalty_keeps_grounded_and_shifts_ungrounded():
    assert gv.penalise(0.8, THETA) == 0.8
    assert gv.penalise(THETA, THETA) == THETA
    assert gv.penalise(0.5, THETA) == pytest.approx(0.5 - THETA)


def test_story_score_is_tanh_of_weighted_mean():
    phrases = [gv.Phrase("a", 2.0, 0.8), gv.Phrase("b", 4.0, 0.5)]
    expected = math.tanh((2.0 * 0.8 + 4.0 * (0.5 - THETA)) / 2)
    assert gv.story_score(phrases, THETA) == pytest.approx(expected)
    assert -1 < gv.story_score(phrases, THETA) < 1


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


def _setup(tmp_path, story_dict, generated_dict, monkeypatch, stories, threshold="{vist: 0.6}"):
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
        chunks={"A dog.": [_tok("a dog")], "It ran.": [[("it", "it", "PRON")]]},
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
            "eval.metrics=[groovist]",
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
    assert float(rows["s1"]) == pytest.approx(math.tanh(3.0 * 2.5 * 0.32))
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
