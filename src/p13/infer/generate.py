"""Generate stories for a dataset split -> GeneratedStory JSONL.

Owner: Behrad. The same code path serves:

* Adapter OFF (zero-shot, E0): ``cfg.model.adapter = null``.
* Adapter ON (fine-tuned, E1+): ``cfg.model.adapter = <path or hub id>``.

Design highlights:

* Pure helpers (:func:`load_split_records`, :func:`build_messages`, :func:`clean_story_text`,
  :func:`build_gen_kwargs`, :func:`gen_config_dict`, :func:`load_existing_stories`) have no
  ``torch`` dependency so they run in CPU-only CI.
* :func:`generate` supports incremental checkpointing and resume via :func:`write_jsonl`,
  batched generation (``cfg.generation.batch_size``), and lazy model loading (if all output
  files are already complete on resume, no model weights are loaded).
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from p13.common.config import ExperimentConfig
from p13.common.jsonl import read_jsonl, write_jsonl
from p13.common.prompts import get_prompt
from p13.common.schema import GeneratedStory, StoryRecord
from p13.common.seed import set_seed
from p13.common.subset import read_ids, subset_path
from p13.infer.model_loading import load_model

log = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


def load_split_records(cfg: ExperimentConfig, dataset: str) -> list[StoryRecord]:
    """Load validated :class:`StoryRecord` items for ``dataset`` and ``cfg.data.split``.

    When ``cfg.data.subset == "dev200"``, filters to the committed dev-subset ID list in
    exact list order and raises if any listed ID is absent from the JSONL file.
    """
    split = cfg.data.split
    jsonl_path = Path(cfg.data.jsonl_dir) / f"{dataset}_{split}.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"{jsonl_path} not found; run p13-prepare or unpack the processed JSONL first"
        )
    records = read_jsonl(jsonl_path, StoryRecord)
    if cfg.data.subset != "dev200":
        return records

    sub_path = subset_path(cfg.data.splits_dir, dataset, split, cfg.data.dev_subset_size)
    if not sub_path.is_file():
        raise FileNotFoundError(
            f"{sub_path} not found; run p13-prepare first or use --set data.subset=full"
        )
    want_ids = read_ids(sub_path)
    by_id = {r.id: r for r in records}
    missing = [i for i in want_ids if i not in by_id]
    if missing:
        raise RuntimeError(
            f"{sub_path} lists {len(missing)} id(s) missing from {jsonl_path} (e.g. {missing[0]})"
        )
    return [by_id[i] for i in want_ids]


def build_messages(image_paths: Sequence[str | Path], prompt_text: str) -> list[dict[str, Any]]:
    """Build one Qwen3-VL chat message with 3 to 10 images in order followed by ``prompt_text``."""
    content: list[dict[str, Any]] = [{"type": "image", "image": str(p)} for p in image_paths]
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def clean_story_text(raw_text: str) -> str:
    """Strip ``<think>...</think>`` blocks and collapse stray whitespace."""
    cleaned = _THINK_BLOCK_RE.sub(" ", raw_text)
    cleaned = _THINK_TAG_RE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def build_gen_kwargs(cfg: ExperimentConfig) -> dict[str, Any]:
    """Build ``model.generate`` keyword arguments from ``cfg.generation``.

    ``temperature`` and ``top_p`` are only included when ``do_sample=True`` so greedy
    decoding does not trigger unused-sampling-flag warnings in Transformers.
    """
    gen = cfg.generation
    kwargs: dict[str, Any] = {
        "max_new_tokens": gen.max_new_tokens,
        "do_sample": gen.do_sample,
        "repetition_penalty": gen.repetition_penalty,
    }
    if gen.do_sample:
        kwargs["temperature"] = gen.temperature
        kwargs["top_p"] = gen.top_p
    return kwargs


def gen_config_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Build the ``gen_config`` metadata dictionary stored in each :class:`GeneratedStory`."""
    return {
        **build_gen_kwargs(cfg),
        "seed": cfg.experiment.seed,
        "max_image_side": cfg.data.max_image_side,
    }


def load_existing_stories(out_path: Path) -> list[GeneratedStory]:
    """Load already-generated stories from ``out_path`` if it exists (for resume)."""
    if not out_path.is_file():
        return []
    existing = read_jsonl(out_path, GeneratedStory)
    log.info("resuming %s: %d stories already generated", out_path, len(existing))
    return existing


def _open_rgb_images(paths: Sequence[str | Path]) -> list[Image.Image]:
    """Open each path and copy as an in-memory RGB image so file handles close immediately."""
    images: list[Image.Image] = []
    for p in paths:
        with Image.open(p) as im:
            images.append(im.convert("RGB"))
    return images


def _inference_context() -> Any:
    try:
        import torch
    except ImportError:
        return contextlib.nullcontext()
    return torch.inference_mode()


def _run_batch(
    batch: Sequence[StoryRecord],
    prompt_text: str,
    model: Any,
    processor: Any,
    gen_kwargs: dict[str, Any],
) -> tuple[list[str], int]:
    """Run one batch of stories through ``processor`` and ``model.generate``."""
    texts: list[str] = []
    flat_images: list[Image.Image] = []
    try:
        for rec in batch:
            messages = build_messages(rec.images, prompt_text)
            chat_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(chat_text)
            flat_images.extend(_open_rgb_images(rec.images))
        inputs = processor(text=texts, images=flat_images, padding=True, return_tensors="pt")
    finally:
        for im in flat_images:
            im.close()

    device = getattr(model, "device", None)
    if device is not None and hasattr(inputs, "to"):
        inputs = inputs.to(device)

    input_ids = inputs["input_ids"]
    input_len = int(input_ids.shape[1])
    total_input_tokens = int(input_ids.numel())

    with _inference_context():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = [out[input_len:] for out in output_ids]
    decoded = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return [clean_story_text(text) for text in decoded], total_input_tokens


def generate(
    cfg: ExperimentConfig,
    *,
    loader: Callable[[ExperimentConfig], tuple[Any, Any]] = load_model,
    save_every: int = 10,
) -> list[Path]:
    """Generate stories for every ``dataset x prompt`` in ``cfg``; return written file paths."""
    if cfg.generation.backend != "transformers":
        raise NotImplementedError(
            f"generation.backend={cfg.generation.backend!r} is not implemented yet; "
            "use generation.backend=transformers"
        )
    if save_every < 1:
        raise ValueError(f"save_every must be >= 1, got {save_every}")

    set_seed(cfg.experiment.seed)
    gen_kwargs = build_gen_kwargs(cfg)
    meta_gen_cfg = gen_config_dict(cfg)
    batch_size = max(1, cfg.generation.batch_size)

    model: Any = None
    processor: Any = None
    written_files: list[Path] = []

    for dataset in cfg.data.datasets:
        records = load_split_records(cfg, dataset)
        for pid in cfg.prompt.ids:
            prompt_text = get_prompt(pid, cfg.prompt.file)
            out_path = cfg.run_dir / f"stories_{dataset}_{cfg.data.split}_{pid}.jsonl"
            existing = load_existing_stories(out_path)
            done_ids = {s.id for s in existing}
            pending = [r for r in records if r.id not in done_ids]

            if not pending:
                log.info(
                    "%s/%s/%s: all %d stories already exist in %s",
                    dataset,
                    cfg.data.split,
                    pid,
                    len(existing),
                    out_path,
                )
                written_files.append(out_path)
                continue

            if model is None:
                model, processor = loader(cfg)

            outputs: list[GeneratedStory] = list(existing)
            new_since_save = 0
            total_tokens = 0
            t0 = time.perf_counter()

            for start in tqdm(
                range(0, len(pending), batch_size),
                desc=f"{dataset}/{cfg.data.split}/{pid}",
                leave=False,
            ):
                batch = pending[start : start + batch_size]
                stories, batch_tokens = _run_batch(batch, prompt_text, model, processor, gen_kwargs)
                total_tokens += batch_tokens
                for rec, story_text in zip(batch, stories, strict=True):
                    outputs.append(
                        GeneratedStory(
                            id=rec.id,
                            dataset=rec.dataset,
                            split=rec.split,
                            model=cfg.model.name,
                            adapter=cfg.model.adapter,
                            prompt_id=pid,
                            story=story_text,
                            gen_config=meta_gen_cfg,
                        )
                    )
                    new_since_save += 1

                if new_since_save >= save_every:
                    write_jsonl(out_path, outputs)
                    new_since_save = 0

            write_jsonl(out_path, outputs)
            elapsed = time.perf_counter() - t0
            n_new = len(pending)
            avg_tokens = total_tokens / n_new
            log.info(
                "%s/%s/%s: generated %d new stories (%d total) -> %s "
                "(%.2fs/story, avg %.0f input tokens/story)",
                dataset,
                cfg.data.split,
                pid,
                n_new,
                len(outputs),
                out_path,
                elapsed / n_new,
                avg_tokens,
            )
            _log_peak_vram()
            written_files.append(out_path)

    return written_files


def _log_peak_vram() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        log.info("peak GPU memory allocated: %.2f GB", torch.cuda.max_memory_allocated() / 1e9)
