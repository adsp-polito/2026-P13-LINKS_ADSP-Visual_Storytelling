"""Generate stories for a dataset split → GeneratedStory JSONL.

**Stub – owner: Behrad.** The same code path serves:

* **Adapter OFF** (zero-shot, E0): ``cfg.model.adapter = null``.
* **Adapter ON** (fine-tuned, E1+): ``cfg.model.adapter = <path or hub id>``.

To implement :func:`generate`:

1. ``set_seed(cfg.experiment.seed)``; load records with
   ``read_jsonl(<jsonl_dir>/<dataset>_<split>.jsonl, StoryRecord)``; if
   ``cfg.data.subset == "dev200"`` keep only ids from ``subset_path(...)``.
2. ``model, processor = load_model(cfg)``.
3. For each record and each prompt id in ``cfg.prompt.ids``: build a chat message with all
   images (in order) followed by ``get_prompt(pid)``; apply the chat template; generate with
   ``cfg.generation``; strip special tokens / thinking tags.
4. Emit ``GeneratedStory(id, dataset, split, model=cfg.model.name, adapter=..., prompt_id=pid,
   story=..., gen_config={...decoding settings + seed...})``.
5. ``write_jsonl(cfg.run_dir / f"stories_{dataset}_{split}_{pid}.jsonl", ...)``.
   Write in batches / resume from an existing file so a crash does not lose a long run.
6. Optional: ``cfg.generation.backend == "vllm"`` for faster batched inference.
"""

from __future__ import annotations

from pathlib import Path

from p13.common.config import ExperimentConfig


def generate(cfg: ExperimentConfig) -> list[Path]:
    """Generate stories for every dataset × prompt in the config; return written files."""
    raise NotImplementedError("generation not implemented yet – see module docstring (Behrad)")
