"""Generate stories for a dataset split -> GeneratedStory JSONL.

Owner: Behrad. The same code path serves:

* Adapter OFF (zero-shot, E0): ``cfg.model.adapter = null``.
* Adapter ON (fine-tuned, E1+): ``cfg.model.adapter = <path or hub id>``.

Design highlights:

* Pure helpers (:func:`load_split_records`, :func:`build_messages`, :func:`clean_story_text`,
  :func:`build_gen_kwargs`, :func:`gen_config_dict`, :func:`is_truncated`,
  :func:`load_existing_stories`) have no ``torch`` dependency so they run in CPU-only CI.
* :func:`generate` validates existing output settings before resuming, warns when any story
  hits ``max_new_tokens`` without an EOS token, supports batched generation, and skips
  loading model weights if all output files are already complete.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from p13.common.config import ExperimentConfig
from p13.common.jsonl import read_jsonl, write_jsonl
from p13.common.prompts import get_prompt
from p13.common.schema import GeneratedStory, StoryRecord
from p13.common.seed import set_seed
from p13.common.subset import load_subset_records
from p13.infer.model_loading import load_model

log = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


def load_split_records(cfg: ExperimentConfig, dataset: str) -> list[StoryRecord]:
    """Load validated :class:`StoryRecord` items for ``dataset`` and ``cfg.data.split``.

    ``data.subset`` is ``full`` or a tag such as ``dev200`` / ``1010``; the selection is
    done by :func:`p13.common.subset.load_subset_records`, which evaluation uses too.
    """
    return load_subset_records(cfg, dataset)


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


def is_truncated(
    token_ids: Sequence[int],
    max_new_tokens: int,
    eos_ids: set[int],
    pad_id: int | None = None,
) -> bool:
    """True if generation hit ``max_new_tokens`` without ending on an EOS token."""
    toks = list(token_ids)
    if pad_id is not None and pad_id not in eos_ids:
        while toks and toks[-1] == pad_id:
            toks.pop()
    if len(toks) < max_new_tokens:
        return False
    return not (eos_ids and toks[-1] in eos_ids)


def _stop_token_ids(model: Any, processor: Any) -> tuple[set[int], int | None]:
    """Extract EOS token IDs and pad token ID from ``model`` and ``processor``."""
    eos_ids: set[int] = set()
    gen_cfg = getattr(model, "generation_config", None)
    tokenizer = getattr(processor, "tokenizer", None)
    for raw in (getattr(gen_cfg, "eos_token_id", None), getattr(tokenizer, "eos_token_id", None)):
        if isinstance(raw, int):
            eos_ids.add(raw)
        elif isinstance(raw, Iterable):
            eos_ids.update(int(x) for x in raw)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(gen_cfg, "pad_token_id", None)
    return eos_ids, int(pad_id) if isinstance(pad_id, int) else None


def load_existing_stories(
    out_path: Path,
    *,
    dataset: str,
    split: str,
    model_name: str,
    adapter: str | None,
    prompt_id: str,
    expected_gen_config: dict[str, Any],
) -> list[GeneratedStory]:
    """Load already-generated stories from ``out_path`` and verify settings match ``cfg``.

    Raises :class:`RuntimeError` if any existing story was produced with a different model,
    adapter, prompt, dataset/split, or ``gen_config`` (for example ``max_new_tokens``).
    """
    if not out_path.is_file():
        return []
    existing = read_jsonl(out_path, GeneratedStory)
    expected = (dataset, split, model_name, adapter, prompt_id, expected_gen_config)
    for s in existing:
        actual = (s.dataset, s.split, s.model, s.adapter, s.prompt_id, s.gen_config)
        if actual != expected:
            raise RuntimeError(
                f"cannot resume {out_path} (story id={s.id!r}): existing settings "
                f"(model={s.model!r}, adapter={s.adapter!r}, prompt_id={s.prompt_id!r}, "
                f"gen_config={s.gen_config}) do not match current config "
                f"(model={model_name!r}, adapter={adapter!r}, prompt_id={prompt_id!r}, "
                f"gen_config={expected_gen_config}); use a new experiment.name or delete {out_path}"
            )
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
    eos_ids: set[int],
    pad_id: int | None,
) -> tuple[list[str], list[bool], int]:
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

    max_new_tokens = int(gen_kwargs["max_new_tokens"])
    new_tokens = [out[input_len:] for out in output_ids]
    truncated = [
        is_truncated(
            tok.tolist() if hasattr(tok, "tolist") else list(tok),
            max_new_tokens,
            eos_ids,
            pad_id,
        )
        for tok in new_tokens
    ]
    decoded = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return [clean_story_text(text) for text in decoded], truncated, total_input_tokens


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
    eos_ids: set[int] = set()
    pad_id: int | None = None
    written_files: list[Path] = []

    for dataset in cfg.data.datasets:
        records = load_split_records(cfg, dataset)
        for pid in cfg.prompt.ids:
            prompt_text = get_prompt(pid, cfg.prompt.file)
            out_path = cfg.run_dir / f"stories_{dataset}_{cfg.data.split}_{pid}.jsonl"
            existing = load_existing_stories(
                out_path,
                dataset=dataset,
                split=cfg.data.split,
                model_name=cfg.model.name,
                adapter=cfg.model.adapter,
                prompt_id=pid,
                expected_gen_config=meta_gen_cfg,
            )
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
                eos_ids, pad_id = _stop_token_ids(model, processor)

            outputs: list[GeneratedStory] = list(existing)
            truncated_ids: list[str] = []
            new_since_save = 0
            total_tokens = 0
            t0 = time.perf_counter()

            for start in tqdm(
                range(0, len(pending), batch_size),
                desc=f"{dataset}/{cfg.data.split}/{pid}",
                leave=False,
            ):
                batch = pending[start : start + batch_size]
                stories, batch_trunc, batch_tokens = _run_batch(
                    batch, prompt_text, model, processor, gen_kwargs, eos_ids, pad_id
                )
                total_tokens += batch_tokens
                for rec, story_text, was_trunc in zip(batch, stories, batch_trunc, strict=True):
                    if was_trunc:
                        truncated_ids.append(rec.id)
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
            if truncated_ids:
                log.warning(
                    "%s/%s/%s: %d/%d stories hit max_new_tokens=%d without EOS and may be "
                    "truncated (e.g. id=%s); consider raising generation.max_new_tokens",
                    dataset,
                    cfg.data.split,
                    pid,
                    len(truncated_ids),
                    n_new,
                    cfg.generation.max_new_tokens,
                    truncated_ids[0],
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
