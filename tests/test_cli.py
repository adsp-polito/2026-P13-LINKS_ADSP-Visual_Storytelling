import json

import pytest

from p13 import cli
from p13.common.jsonl import write_jsonl
from p13.common.schema import GeneratedStory

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
        (cli.prepare_main, ["--set", "data.datasets=[aesop]"]),
    ],
    ids=["generate-vllm", "train", "prepare-aesop"],
)
def test_stubs_exit_cleanly(main, extra):
    assert main(["--config", "configs/e0_zeroshot.yaml", *extra]) == 2


def test_eval_validates_input_then_hits_stub(tmp_path, generated_dict):
    path = tmp_path / "stories.jsonl"
    write_jsonl(path, [GeneratedStory.model_validate(generated_dict)])
    assert cli.eval_main(["--config", "configs/e0_zeroshot.yaml", "--stories", str(path)]) == 2
