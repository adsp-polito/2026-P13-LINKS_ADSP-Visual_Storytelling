"""Load the frozen Qwen3-VL base model, optionally 4-bit, optionally with a LoRA adapter.

Owner: Behrad. Used by both generation and training, so the GPU switch lives in one place.

Two layers:

* :func:`plan_load` is pure: config + hardware facts in, a :class:`LoadPlan` out. No torch,
  no downloads, so it runs in CPU-only CI and is easy to test.
* :func:`load_model` reads the real hardware, builds the plan, and does the heavy loading.

Hardware rules (see :func:`plan_load`):

* 4-bit (bitsandbytes NF4) only works on CUDA. On MPS / CPU it is turned off with a warning.
* ``bfloat16`` on a GPU without bf16 support (e.g. Colab T4) falls back to ``float16``.
* CPU always uses ``float32``; MPS uses ``float16``.

``HF_TOKEN`` is read from the environment (``.env`` is loaded by ``setup_logging``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from p13.common.config import ExperimentConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadPlan:
    """Everything :func:`load_model` needs to know, decided before any weights load."""

    model_name: str
    device: str  # "cuda" | "mps" | "cpu"
    dtype: str  # "bfloat16" | "float16" | "float32"
    load_in_4bit: bool
    attn_implementation: str
    max_pixels: int  # processor cap: images are pre-resized to long side <= max_image_side
    adapter: str | None


def plan_load(
    cfg: ExperimentConfig,
    adapter: str | None = None,
    *,
    cuda: bool,
    mps: bool = False,
    bf16_supported: bool = True,
) -> LoadPlan:
    """Decide device, dtype and quantization from the config and the hardware facts.

    ``adapter`` overrides ``cfg.model.adapter`` when given.
    """
    device = "cuda" if cuda else "mps" if mps else "cpu"

    load_in_4bit = cfg.model.load_in_4bit
    if load_in_4bit and device != "cuda":
        log.warning("model.load_in_4bit needs CUDA; loading unquantized on %s instead", device)
        load_in_4bit = False

    dtype = cfg.model.dtype
    if device == "cpu":
        dtype = "float32"
    elif device == "mps":
        dtype = "float16"
    elif dtype == "bfloat16" and not bf16_supported:
        log.warning("this GPU has no bfloat16 support; using float16")
        dtype = "float16"

    side = cfg.data.max_image_side
    return LoadPlan(
        model_name=cfg.model.name,
        device=device,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        attn_implementation=cfg.model.attn_implementation,
        max_pixels=side * side,
        adapter=adapter if adapter is not None else cfg.model.adapter,
    )


def load_model(cfg: ExperimentConfig, adapter: str | None = None) -> tuple[Any, Any]:
    """Return ``(model, processor)`` in eval mode. ``adapter`` overrides ``cfg.model.adapter``."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    cuda = torch.cuda.is_available()
    plan = plan_load(
        cfg,
        adapter,
        cuda=cuda,
        mps=torch.backends.mps.is_available(),
        bf16_supported=cuda and torch.cuda.is_bf16_supported(),
    )
    log.info(
        "loading %s on %s (dtype=%s, 4bit=%s, adapter=%s)",
        plan.model_name,
        plan.device,
        plan.dtype,
        plan.load_in_4bit,
        plan.adapter,
    )
    token = os.environ.get("HF_TOKEN") or None
    dtype = getattr(torch, plan.dtype)

    processor = AutoProcessor.from_pretrained(
        plan.model_name, max_pixels=plan.max_pixels, token=token
    )
    # Left padding so batched prompts end where generation starts.
    processor.tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "attn_implementation": plan.attn_implementation,
        "token": token,
    }
    if plan.device == "cuda":
        kwargs["device_map"] = "auto"
    if plan.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(plan.model_name, **kwargs)
    if plan.device != "cuda":
        model = model.to(plan.device)

    if plan.adapter:
        from peft import PeftModel

        log.info("attaching LoRA adapter %s", plan.adapter)
        model = PeftModel.from_pretrained(model, plan.adapter)

    model.eval()
    if cuda:
        log.info("model loaded; GPU memory allocated: %.2f GB", torch.cuda.memory_allocated() / 1e9)
    return model, processor
