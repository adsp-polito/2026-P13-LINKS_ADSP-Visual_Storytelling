"""LoRA / QLoRA supervised fine-tuning on SIS stories.

**Stub – owner: Behrad.**

Requirements (from the project design, Zone 1):

* **Frozen base.** Vision encoder and language decoder are frozen; only LoRA adapters train
  (``cfg.lora.target_modules``; vision tower untouched unless ``cfg.lora.train_vision_tower``).
* **QLoRA on small GPUs.** ``cfg.model.load_in_4bit`` → 4-bit NF4 base +
  ``peft.prepare_model_for_kbit_training``; otherwise bf16 LoRA.
* **Masked-prompt loss.** Cross-entropy only on the story (assistant) tokens. Image and prompt
  tokens get label ``-100``. Use trl's ``SFTTrainer`` with assistant-only loss, or a custom
  collator that masks everything up to the assistant turn.
* **Training input.** ``StoryRecord`` from ``<jsonl_dir>/vist_train.jsonl`` (SIS ``story`` as
  target), images in order + ``get_prompt(cfg.prompt.train_id)`` as the user turn.
* **Budget knobs.** ``cfg.train.per_device_batch_size``, ``grad_accum``,
  ``gradient_checkpointing``, ``max_seq_len`` and ``cfg.data.max_image_side``.
* **Repro / tracking.** ``set_seed(cfg.experiment.seed)``; ``init_wandb(cfg, "train")`` if
  enabled; save the resolved config next to the adapter.
* **Output.** Adapter saved to ``cfg.run_dir / "adapter"``; if ``cfg.hub.push_adapter``, push
  to ``cfg.hub.repo_id`` with ``private=True``. Never commit adapter weights.
"""

from __future__ import annotations

from pathlib import Path

from p13.common.config import ExperimentConfig


def train(cfg: ExperimentConfig) -> Path:
    """Train a LoRA adapter; return the local adapter directory."""
    raise NotImplementedError("LoRA/QLoRA training not implemented yet – see docstring (Behrad)")
