"""Tests for the load plan (pure logic, no torch, no model download)."""

import logging

from p13.common.config import parse_config_dict
from p13.infer.model_loading import plan_load


def _cfg(**model):
    return parse_config_dict({"model": model, "data": {"max_image_side": 448}})


def test_colab_gpu_bf16_unquantized():
    plan = plan_load(_cfg(name="Qwen/Qwen3-VL-4B-Instruct"), cuda=True)
    assert plan.device == "cuda"
    assert plan.dtype == "bfloat16"
    assert plan.load_in_4bit is False
    assert plan.model_name == "Qwen/Qwen3-VL-4B-Instruct"


def test_4bit_kept_on_cuda():
    assert plan_load(_cfg(load_in_4bit=True), cuda=True).load_in_4bit is True


def test_4bit_turned_off_without_cuda(caplog):
    with caplog.at_level(logging.WARNING):
        plan = plan_load(_cfg(load_in_4bit=True), cuda=False)
    assert plan.load_in_4bit is False
    assert plan.device == "cpu"
    assert "needs CUDA" in caplog.text


def test_gpu_without_bf16_falls_back_to_float16():
    plan = plan_load(_cfg(dtype="bfloat16"), cuda=True, bf16_supported=False)
    assert plan.dtype == "float16"


def test_cpu_uses_float32_and_mps_uses_float16():
    assert plan_load(_cfg(), cuda=False).dtype == "float32"
    mps = plan_load(_cfg(), cuda=False, mps=True)
    assert (mps.device, mps.dtype) == ("mps", "float16")


def test_max_pixels_matches_image_side():
    assert plan_load(_cfg(), cuda=True).max_pixels == 448 * 448


def test_adapter_argument_overrides_config():
    cfg = _cfg(adapter="outputs/E1/adapter")
    assert plan_load(cfg, cuda=True).adapter == "outputs/E1/adapter"
    assert plan_load(cfg, "hub/other", cuda=True).adapter == "hub/other"
    assert plan_load(_cfg(), cuda=True).adapter is None
