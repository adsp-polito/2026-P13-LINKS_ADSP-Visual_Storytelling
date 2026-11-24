import csv
import json

import pytest

from p13 import cli
from p13.common.jsonl import write_jsonl
from p13.common.schema import GeneratedStory, StoryRecord

MAINS = [cli.prepare_main, cli.generate_main, cli.train_main, cli.eval_main]


@pytest.mark.parametrize("main", MAINS, ids=lambda f: f.__name__)
def test_help(main, capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_dry_run_prints_resolved_config(capsys):
    rc = cli.train_main(
        ["--config", "configs/e1_lora_2b_6gb.yaml", "--set", "train.grad_accum=4", "--dry-run"]
    )
    assert rc == 0
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["train"]["grad_accum"] == 4 and cfg["model"]["load_in_4bit"] is True


@pytest.mark.parametrize(
    "main,extra",
    [
        (cli.generate_main, ["--set", "generation.backend=vllm"]),
        (cli.train_main, []),
        (cli.prepare_main, ["--set", "data.datasets=[vwp]"]),
    ],
    ids=["generate-vllm", "train", "prepare-vwp"],
)
def test_stubs_exit_cleanly(main, extra):
    assert main(["--config", "configs/e0_zeroshot.yaml", *extra]) == 2


def _eval_args(tmp_path, story_path, metric_names):
    return [
        "--config",
        "configs/e0_zeroshot.yaml",
        "--stories",
        str(story_path),
        "--set",
        f"data.jsonl_dir={tmp_path / 'processed'}",
        "--set",
        f"output.dir={tmp_path / 'outputs'}",
        "--set",
        f"eval.metrics=[{metric_names}]",
    ]


def test_eval_writes_scores(tmp_path, story_dict, generated_dict):
    write_jsonl(tmp_path / "processed" / "vist_val.jsonl", [StoryRecord(**story_dict)])
    path = tmp_path / "stories.jsonl"
    write_jsonl(path, [GeneratedStory.model_validate(generated_dict)])
    assert cli.eval_main(_eval_args(tmp_path, path, "length,distinct_2")) == 0
    assert list((tmp_path / "outputs").glob("*/scores.csv"))


def test_eval_stub_metric_exits_cleanly(tmp_path, story_dict, generated_dict):
    write_jsonl(tmp_path / "processed" / "vist_val.jsonl", [StoryRecord(**story_dict)])
    path = tmp_path / "stories.jsonl"
    write_jsonl(path, [GeneratedStory.model_validate(generated_dict)])
    assert cli.eval_main(_eval_args(tmp_path, path, "length,groovist")) == 2


def _write_one(tmp_path, story_dict, generated_dict, split="val", **gen_fields):
    ref = StoryRecord(**{**story_dict, "split": split})
    write_jsonl(tmp_path / "processed" / f"vist_{split}.jsonl", [ref])
    path = tmp_path / "stories.jsonl"
    gen = GeneratedStory.model_validate({**generated_dict, "split": split, **gen_fields})
    write_jsonl(path, [gen])
    return path


def test_eval_refuses_test_split_by_default(tmp_path, story_dict, generated_dict):
    path = _write_one(tmp_path, story_dict, generated_dict, split="test")
    with pytest.raises(ValueError, match="test split"):
        cli.eval_main(_eval_args(tmp_path, path, "length"))
    assert not list((tmp_path / "outputs").glob("*/scores.csv"))


def test_eval_scores_test_split_when_allowed(tmp_path, story_dict, generated_dict):
    path = _write_one(tmp_path, story_dict, generated_dict, split="test")
    args = [*_eval_args(tmp_path, path, "length"), "--set", "eval.allow_test=true"]
    assert cli.eval_main(args) == 0
    assert list((tmp_path / "outputs").glob("*/scores.csv"))


def test_eval_human_baseline_skips_reference_metrics(tmp_path, story_dict, generated_dict):
    path = _write_one(
        tmp_path, story_dict, generated_dict, model="human", adapter=None, prompt_id="reference"
    )
    assert cli.eval_main(_eval_args(tmp_path, path, "length,bleu4")) == 0
    (scores,) = (tmp_path / "outputs").glob("*/scores.csv")
    with scores.open(encoding="utf-8", newline="") as f:
        (row,) = csv.DictReader(f)
    assert row["model"] == "human"
    assert row["length"] != ""
    assert row["bleu4"] == ""


def test_eval_with_human_flag_scores_the_references(tmp_path, story_dict, generated_dict):
    path = _write_one(tmp_path, story_dict, generated_dict)
    assert cli.eval_main([*_eval_args(tmp_path, path, "length"), "--with-human"]) == 0
    (scores,) = (tmp_path / "outputs").glob("*/scores.csv")
    with scores.open(encoding="utf-8", newline="") as f:
        models = [row["model"] for row in csv.DictReader(f)]
    assert sorted(models) == ["Qwen/Qwen3-VL-2B-Instruct", "human"]


def test_eval_checks_metric_names_before_reading_stories(tmp_path):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="unknown metric"):
        cli.eval_main(_eval_args(tmp_path, missing, "length,nope"))
