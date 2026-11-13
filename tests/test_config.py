from pathlib import Path

import pytest

from p13.common.config import ConfigError, load_config
from p13.common.prompts import get_prompt

CONFIGS = sorted(p for p in Path("configs").glob("e*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_example_configs_load(path):
    cfg = load_config(path)
    assert cfg.experiment.seed == 42
    assert cfg.data.max_image_side == 448
    for pid in cfg.prompt.ids:
        assert get_prompt(pid)


def test_gpu_profiles_differ_as_intended():
    small = load_config("configs/e1_lora_2b_6gb.yaml")
    big = load_config("configs/e1_lora_4b_24gb.yaml")
    assert small.model.load_in_4bit and not big.model.load_in_4bit
    assert "2B" in small.model.name and "4B" in big.model.name
    assert small.effective_batch_size == big.effective_batch_size == 16


def test_overrides():
    cfg = load_config(
        "configs/e1_lora_2b_6gb.yaml",
        [
            "model.name=Qwen/Qwen3-VL-8B-Instruct",
            "model.load_in_4bit=false",
            "train.grad_accum=8",
            "data.max_image_side=336",
        ],
    )
    assert cfg.model.name.endswith("8B-Instruct")
    assert cfg.model.load_in_4bit is False
    assert cfg.train.grad_accum == 8
    assert cfg.data.max_image_side == 336


@pytest.mark.parametrize(
    "override",
    [
        "model.nmae=x",
        "model.load_in_4bit=yes-please",
        "data.datasets=[coco]",
        "data.subset=half",
        "no_equals_sign",
    ],
)
def test_bad_values_rejected(override):
    with pytest.raises(ConfigError):
        load_config("configs/e0_zeroshot.yaml", [override])


def test_prompts_text():
    assert get_prompt("P0") == "Write a coherent story from this sequence of images."
    p1 = get_prompt("P1")
    assert p1.startswith(get_prompt("P0")) and "50–100 words" in p1
    with pytest.raises(KeyError):
        get_prompt("P99")
