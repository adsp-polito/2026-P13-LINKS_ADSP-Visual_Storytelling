"""Load the frozen Qwen3-VL base model, optionally 4-bit, optionally with a LoRA adapter.

**Stub – owner: Behrad.** Used by both generation and training, so the 6 GB / 24 GB switch
lives in exactly one place.

To implement :func:`load_model`:

* ``AutoProcessor.from_pretrained(cfg.model.name)``. Set the processor's ``max_pixels`` so
  the long side matches ``cfg.data.max_image_side`` (images are pre-resized, but keep both in sync).
* ``Qwen3VLForConditionalGeneration.from_pretrained`` (or ``AutoModelForImageTextToText``) with
  ``torch_dtype=cfg.model.dtype``, ``attn_implementation=cfg.model.attn_implementation``,
  ``device_map="auto"``.
* If ``cfg.model.load_in_4bit``: pass ``BitsAndBytesConfig(load_in_4bit=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=<dtype>)``.
* If an ``adapter`` is given: ``PeftModel.from_pretrained(model, adapter)`` (inference only).
* ``model.eval()`` for inference; training wraps the model itself in ``train/sft.py``.
* Read ``HF_TOKEN`` from the environment (``.env``); never hard-code it.
"""

from __future__ import annotations

from typing import Any

from p13.common.config import ExperimentConfig


def load_model(cfg: ExperimentConfig, adapter: str | None = None) -> tuple[Any, Any]:
    """Return ``(model, processor)``. ``adapter`` overrides ``cfg.model.adapter``."""
    raise NotImplementedError("model loading not implemented yet – see module docstring (Behrad)")
