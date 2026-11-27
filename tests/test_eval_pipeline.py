"""evaluate(): joins generated stories to references and writes per-item scores.csv."""

from __future__ import annotations

import csv
import math

import pytest

from p13.common.config import ExperimentConfig, load_config
from p13.common.jsonl import write_jsonl
from p13.common.schema import GeneratedStory, StoryRecord
from p13.eval import metrics
from p13.eval.metrics import GROUP_COLUMNS, ID_COLUMNS, distinct_2, evaluate, length


def _cfg(tmp_path, metric_names):
    names = ",".join(metric_names)
    return load_config(
        "configs/e0_zeroshot.yaml",
        [
            f"data.jsonl_dir={tmp_path / 'processed'}",
            f"output.dir={tmp_path / 'outputs'}",
            f"eval.metrics=[{names}]",
        ],
    )


def _refs(tmp_path, story_dict, ids=("s1", "s2")):
    recs = [StoryRecord.model_validate({**story_dict, "id": i}) for i in ids]
    write_jsonl(tmp_path / "processed" / "vist_val.jsonl", recs)


def _gens(tmp_path, generated_dict, name="gens.jsonl", ids=("s1", "s2"), **fields):
    recs = [GeneratedStory.model_validate({**generated_dict, "id": i, **fields}) for i in ids]
    path = tmp_path / name
    write_jsonl(path, recs)
    return path


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _g(generated_dict, story):
    return GeneratedStory.model_validate({**generated_dict, "story": story})


def test_length_counts_words_not_punctuation(generated_dict):
    key = ("vist", "val", "s1")
    assert (
        length([_g(generated_dict, "The dog ran . It's happy!")], {}, ExperimentConfig())[key]
        == 5.0
    )
    assert length([_g(generated_dict, "")], {}, ExperimentConfig())[key] == 0.0


def test_distinct_2_is_over_all_stories(generated_dict):
    assert distinct_2([_g(generated_dict, "a b a b")]) == pytest.approx(2 / 3)
    same = [_g(generated_dict, "a dog ran"), _g({**generated_dict, "id": "s2"}, "A dog ran.")]
    assert distinct_2(same) == pytest.approx(2 / 4)
    assert distinct_2([_g(generated_dict, "x ran"), _g(generated_dict, "a y")]) == 1.0
    assert distinct_2([_g(generated_dict, "dog"), _g(generated_dict, "")]) == 0.0


def test_evaluate_writes_one_row_per_story(tmp_path, story_dict, generated_dict):
    _refs(tmp_path, story_dict)
    gens = _gens(tmp_path, generated_dict)
    cfg = _cfg(tmp_path, ["length", "distinct_2"])

    out = evaluate(cfg, [gens])

    assert out == cfg.run_dir / "scores.csv"
    rows = _read_csv(out)
    assert list(rows[0]) == [*ID_COLUMNS, "length"]
    assert [r["id"] for r in rows] == ["s1", "s2"]
    assert rows[0]["adapter"] == ""
    assert float(rows[0]["length"]) == 14.0
    assert not (cfg.run_dir / "scores.csv.tmp").exists()

    groups = _read_csv(cfg.run_dir / "system_scores.csv")
    assert list(groups[0]) == [*GROUP_COLUMNS, "distinct_2"]
    assert len(groups) == 1 and groups[0]["n_stories"] == "2"
    assert float(groups[0]["distinct_2"]) == pytest.approx(0.5)


def test_evaluate_without_corpus_metrics_removes_old_system_file(
    tmp_path, story_dict, generated_dict
):
    _refs(tmp_path, story_dict)
    gens = _gens(tmp_path, generated_dict)
    evaluate(_cfg(tmp_path, ["length", "distinct_2"]), [gens])
    cfg = _cfg(tmp_path, ["length"])
    assert (cfg.run_dir / "system_scores.csv").exists()

    evaluate(cfg, [gens])
    assert not (cfg.run_dir / "system_scores.csv").exists()


def test_evaluate_never_pools_datasets(tmp_path, story_dict, generated_dict, monkeypatch):
    _refs(tmp_path, story_dict)
    write_jsonl(
        tmp_path / "processed" / "aesop_val.jsonl",
        [StoryRecord.model_validate({**story_dict, "dataset": "aesop"})],
    )
    vist = _gens(tmp_path, generated_dict, "vist.jsonl")
    aesop = _gens(tmp_path, generated_dict, "aesop.jsonl", ids=("s1",), dataset="aesop")
    calls = []

    def fake(gens, refs, cfg):
        calls.append({g.dataset for g in gens})
        return {g.key: 1.0 for g in gens}

    monkeypatch.setitem(metrics.METRICS, "length", fake)
    cfg = _cfg(tmp_path, ["length", "distinct_2"])
    evaluate(cfg, [vist, aesop])

    assert calls == [{"aesop"}, {"vist"}]
    groups = _read_csv(cfg.run_dir / "system_scores.csv")
    assert [(r["dataset"], r["n_stories"]) for r in groups] == [("aesop", "1"), ("vist", "2")]


def test_evaluate_keeps_systems_apart(tmp_path, story_dict, generated_dict, monkeypatch):
    _refs(tmp_path, story_dict)
    zero = _gens(tmp_path, generated_dict, "zero.jsonl")
    lora = _gens(tmp_path, generated_dict, "lora.jsonl", adapter="outputs/E1/adapter")
    calls = []

    def fake(gens, refs, cfg):
        calls.append({g.adapter for g in gens})
        return {g.key: 1.0 for g in gens}

    monkeypatch.setitem(metrics.METRICS, "length", fake)
    rows = _read_csv(evaluate(_cfg(tmp_path, ["length"]), [zero, lora]))

    assert len(rows) == 4
    assert {r["adapter"] for r in rows} == {"", "outputs/E1/adapter"}
    assert calls == [{None}, {"outputs/E1/adapter"}]


def test_evaluate_fails_if_a_story_has_no_reference(tmp_path, story_dict, generated_dict):
    _refs(tmp_path, story_dict, ids=("s1",))
    gens = _gens(tmp_path, generated_dict, ids=("s1", "s2"))
    with pytest.raises(ValueError, match=r"1 of 2 stories .* no reference"):
        evaluate(_cfg(tmp_path, ["length"]), [gens])


def test_evaluate_fails_without_reference_file(tmp_path, generated_dict):
    gens = _gens(tmp_path, generated_dict)
    with pytest.raises(FileNotFoundError, match=r"vist_val\.jsonl"):
        evaluate(_cfg(tmp_path, ["length"]), [gens])


def test_evaluate_fails_on_duplicate_story_across_files(tmp_path, story_dict, generated_dict):
    _refs(tmp_path, story_dict)
    a = _gens(tmp_path, generated_dict, "a.jsonl")
    b = _gens(tmp_path, generated_dict, "b.jsonl")
    with pytest.raises(ValueError, match="appears twice"):
        evaluate(_cfg(tmp_path, ["length"]), [a, b])


def test_evaluate_fails_on_unknown_metric_before_reading(tmp_path):
    with pytest.raises(ValueError, match="unknown metric"):
        evaluate(_cfg(tmp_path, ["length", "bleu5"]), [tmp_path / "does_not_exist.jsonl"])


def test_evaluate_leaves_cell_empty_for_missing_score(
    tmp_path, story_dict, generated_dict, monkeypatch
):
    _refs(tmp_path, story_dict)
    gens = _gens(tmp_path, generated_dict)

    def partial(gens, refs, cfg):
        return {gens[0].key: 0.5, gens[1].key: math.nan}

    monkeypatch.setitem(metrics.METRICS, "length", partial)
    rows = _read_csv(evaluate(_cfg(tmp_path, ["length"]), [gens]))
    assert [r["length"] for r in rows] == ["0.5", ""]


def test_evaluate_stub_metric_raises_not_implemented(tmp_path, story_dict, generated_dict):
    _refs(tmp_path, story_dict)
    gens = _gens(tmp_path, generated_dict)
    with pytest.raises(NotImplementedError):
        evaluate(_cfg(tmp_path, ["length", "bleu4"]), [gens])


def test_metrics_have_the_evaluation_md_names():
    new = {"halluc_any", "halluc_ner", "halluc_obj_share", "n_objects"}
    assert set(metrics.METRICS) >= new
    assert "hallucination_rate" not in metrics.METRICS
    assert set(metrics.DERIVED_METRICS) == {"dhm_g", "dhm_c", "dhm_nr", "dhm"}
    assert not set(metrics.METRICS) & set(metrics.DERIVED_METRICS)
    assert set(metrics.METRICS) | set(metrics.DERIVED_METRICS) >= metrics.REFERENCE_METRICS


def test_default_config_metrics_all_exist():
    cfg = load_config("configs/e0_zeroshot.yaml")
    known = set(metrics.METRICS) | set(metrics.DERIVED_METRICS) | set(metrics.CORPUS_METRICS)
    assert set(cfg.eval.metrics) <= known


K1, K2 = ("vist", "val", "s1"), ("vist", "val", "s2")


def test_derive_dhm_signed_gaps_and_mean_absolute_gap():
    model = {"groovist": {K1: 0.5}, "rovist_c": {K1: 0.25}, "rovist_nr": {K1: 0.9}}
    human = {"groovist": {K1: -0.1}, "rovist_c": {K1: 0.75}, "rovist_nr": {K1: 0.9}}
    out = metrics.derive_dhm([K1], model, human)
    assert out["dhm_g"][K1] == pytest.approx(0.6)
    assert out["dhm_c"][K1] == pytest.approx(-0.5)
    assert out["dhm_nr"][K1] == pytest.approx(0.0)
    assert out["dhm"][K1] == pytest.approx((0.6 + 0.5 + 0.0) / 3)


def test_derive_dhm_missing_side_gives_nan_not_a_guess():
    model = {"groovist": {K1: 0.5, K2: 0.5}, "rovist_c": {K1: 0.5}, "rovist_nr": {K1: 0.5, K2: 0.5}}
    human = {"groovist": {K1: 0.1}, "rovist_c": {K1: 0.1, K2: 0.1}, "rovist_nr": {K2: 0.1}}
    out = metrics.derive_dhm([K1, K2], model, human)
    assert math.isnan(out["dhm_nr"][K1])
    assert math.isnan(out["dhm_c"][K2])
    assert math.isnan(out["dhm"][K1]) and math.isnan(out["dhm"][K2])
    assert out["dhm_g"][K1] == pytest.approx(0.4)


DHM_NAMES = ["groovist", "rovist_c", "rovist_nr", "dhm_g", "dhm_c", "dhm_nr", "dhm"]


def _fake_base_metrics(monkeypatch, calls=None):
    """groovist / rovist_c / rovist_nr = word count / 100, so scores differ by story text."""

    def fake(gens, refs, cfg):
        if calls is not None:
            calls.append(gens[0].model)
        return {g.key: len(g.story.split()) / 100 for g in gens}

    for name in ("groovist", "rovist_c", "rovist_nr"):
        monkeypatch.setitem(metrics.METRICS, name, fake)


def _human(tmp_path, generated_dict, ids=("s1", "s2"), name="human.jsonl", **fields):
    story = "A dog on a beach."
    return _gens(
        tmp_path,
        generated_dict,
        name,
        ids=ids,
        model="human",
        adapter=None,
        prompt_id="reference",
        story=story,
        gen_config={},
        **fields,
    )


def test_evaluate_derives_dhm_against_human_rows(tmp_path, story_dict, generated_dict, monkeypatch):
    _refs(tmp_path, story_dict)
    _fake_base_metrics(monkeypatch)
    model = _gens(tmp_path, generated_dict)
    rows = _read_csv(evaluate(_cfg(tmp_path, DHM_NAMES), [model, _human(tmp_path, generated_dict)]))

    by_model = {(r["model"], r["id"]): r for r in rows}
    m = by_model[("Qwen/Qwen3-VL-2B-Instruct", "s1")]
    assert float(m["dhm_g"]) == pytest.approx(0.14 - 0.05)
    assert float(m["dhm"]) == pytest.approx(0.09)
    h = by_model[("human", "s1")]
    assert float(h["groovist"]) == pytest.approx(0.05)
    assert all(h[n] == "" for n in ("dhm_g", "dhm_c", "dhm_nr", "dhm"))


def test_evaluate_dhm_without_human_stories_fails_before_scoring(
    tmp_path, story_dict, generated_dict, monkeypatch
):
    _refs(tmp_path, story_dict)
    calls = []
    _fake_base_metrics(monkeypatch, calls)
    with pytest.raises(ValueError, match="needs exactly one model='human' group"):
        evaluate(_cfg(tmp_path, DHM_NAMES), [_gens(tmp_path, generated_dict)])
    assert calls == []


def test_evaluate_dhm_without_base_metrics_fails(tmp_path, story_dict, generated_dict):
    _refs(tmp_path, story_dict)
    files = [_gens(tmp_path, generated_dict), _human(tmp_path, generated_dict)]
    with pytest.raises(ValueError, match=r"dhm\* needs \['rovist_nr'\]"):
        evaluate(_cfg(tmp_path, ["groovist", "rovist_c", "dhm"]), files)


def test_evaluate_dhm_fails_if_human_misses_a_story(
    tmp_path, story_dict, generated_dict, monkeypatch
):
    _refs(tmp_path, story_dict)
    calls = []
    _fake_base_metrics(monkeypatch, calls)
    files = [_gens(tmp_path, generated_dict), _human(tmp_path, generated_dict, ids=("s1",))]
    with pytest.raises(ValueError, match="1 of 2 stories have no human story"):
        evaluate(_cfg(tmp_path, DHM_NAMES), files)
    assert calls == []


def test_evaluate_halluc_obj_share_needs_n_objects(tmp_path):
    with pytest.raises(ValueError, match="halluc_obj_share needs n_objects"):
        evaluate(_cfg(tmp_path, ["length", "halluc_obj_share"]), [tmp_path / "none.jsonl"])


def _nr(generated_dict, story):
    return metrics.rovist_nr([_g(generated_dict, story)], {}, ExperimentConfig()).get(
        ("vist", "val", "s1")
    )


def test_rovist_nr_no_repetition_scores_one(generated_dict):
    assert _nr(generated_dict, "A dog runs. The cat sleeps.") == 1.0


def test_rovist_nr_repeated_sentences(generated_dict):
    assert _nr(generated_dict, "A dog runs. A dog runs.") == 0.0
    assert _nr(generated_dict, "a dog ran . a cat ran .") == pytest.approx(1 - 2 / 4)


def test_rovist_nr_ignores_case_and_punctuation(generated_dict):
    assert _nr(generated_dict, "The dog ran! THE DOG RAN?") == _nr(
        generated_dict, "the dog ran . the dog ran ."
    )


def test_rovist_nr_within_sentence_blocks(generated_dict):
    story = "one two three four one two three four. Something else."
    between = 0.0
    within = 1.0
    assert _nr(generated_dict, story) == pytest.approx(1 - (between + within) / 2)


def test_rovist_nr_ignores_trailing_partial_block(generated_dict):
    story = "a b c d a b c d x. Something else."
    assert _nr(generated_dict, story) == pytest.approx(1 - (0.0 + 1.0) / 2)


def test_rovist_nr_short_sentences_have_no_within_part(generated_dict):
    story = "a b c d a. Something else."
    between = 0.0
    assert _nr(generated_dict, story) == pytest.approx(1 - between)


def test_sentences_split_after_closing_quotes_and_brackets():
    text = (
        'He yelled "ROCK SHOWING!" Next came the crowd. (They cheered.) Then \u201cgo.\u201d End.'
    )
    assert metrics._sentences(text) == [
        ["he", "yelled", "rock", "showing"],
        ["next", "came", "the", "crowd"],
        ["they", "cheered"],
        ["then", "go"],
        ["end"],
    ]


def test_rovist_nr_quoted_dialogue_counts_as_sentences(generated_dict):
    assert _nr(generated_dict, '"A dog runs!" A dog runs.') == 0.0


def test_write_csv_removes_tmp_file_on_failure(tmp_path):
    out = tmp_path / "scores.csv"
    with pytest.raises(ValueError):
        metrics._write_csv(out, ["a"], [{"a": 1, "b": 2}])
    assert not out.exists()
    assert not (tmp_path / "scores.csv.tmp").exists()


def test_evaluate_warns_when_replacing_scores(tmp_path, story_dict, generated_dict, caplog):
    _refs(tmp_path, story_dict)
    gens = _gens(tmp_path, generated_dict)
    cfg = _cfg(tmp_path, ["length"])
    evaluate(cfg, [gens])
    assert "replacing" not in caplog.text
    evaluate(cfg, [gens])
    assert "replacing" in caplog.text


def test_rovist_nr_single_sentence_or_empty_has_no_score(generated_dict):
    assert _nr(generated_dict, "Just one sentence about a dog.") is None
    assert _nr(generated_dict, "") is None


def test_rovist_nr_stays_in_unit_range(generated_dict):
    story = "the dog the dog the dog the dog. the dog the dog the dog."
    assert 0.0 <= _nr(generated_dict, story) <= 1.0


def test_human_stories_copy_the_references(story_dict):
    refs = {K1: StoryRecord.model_validate(story_dict)}
    (h,) = metrics.human_stories([K1, K1], refs)
    assert (h.model, h.adapter, h.prompt_id) == ("human", None, "reference")
    assert h.key == K1 and h.story == story_dict["story"]


def test_evaluate_with_human_adds_one_human_group(
    tmp_path, story_dict, generated_dict, monkeypatch
):
    _refs(tmp_path, story_dict)
    zero = _gens(tmp_path, generated_dict, "zero.jsonl", ids=("s1",))
    lora = _gens(tmp_path, generated_dict, "lora.jsonl", adapter="outputs/E1/adapter")
    called_for = []

    def fake_bleu4(gens, refs, cfg):
        called_for.extend(g.model for g in gens)
        return {g.key: 0.5 for g in gens}

    monkeypatch.setitem(metrics.METRICS, "bleu4", fake_bleu4)
    cfg = _cfg(tmp_path, ["length", "bleu4"])
    rows = _read_csv(evaluate(cfg, [zero, lora], with_human=True))

    assert "human" not in called_for

    human = [r for r in rows if r["model"] == "human"]
    assert [r["id"] for r in human] == ["s1", "s2"]
    assert {(r["adapter"], r["prompt_id"]) for r in human} == {("", "reference")}
    assert float(human[0]["length"]) == 15.0
    assert all(r["bleu4"] == "" for r in human)


def test_evaluate_with_human_derives_dhm(tmp_path, story_dict, generated_dict, monkeypatch):
    _refs(tmp_path, story_dict)
    _fake_base_metrics(monkeypatch)
    rows = _read_csv(
        evaluate(_cfg(tmp_path, DHM_NAMES), [_gens(tmp_path, generated_dict)], with_human=True)
    )
    model = next(r for r in rows if r["model"] != "human")
    assert float(model["dhm_g"]) == pytest.approx(0.14 - 0.15)


def test_evaluate_with_human_refuses_human_story_files(tmp_path, story_dict, generated_dict):
    _refs(tmp_path, story_dict)
    files = [_gens(tmp_path, generated_dict), _human(tmp_path, generated_dict)]
    with pytest.raises(ValueError, match="already contain model='human'"):
        evaluate(_cfg(tmp_path, ["length"]), files, with_human=True)
