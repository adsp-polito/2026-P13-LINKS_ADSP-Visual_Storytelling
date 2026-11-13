"""The unified JSONL contract.

Every module reads and writes these two record types, one JSON object per line:

* :class:`StoryRecord` is a reference story produced by the data loaders (``src/p13/data``).
* :class:`GeneratedStory` is a story produced by a model (``src/p13/infer``), consumed by
  ``src/p13/eval``. Evaluation joins on ``(dataset, split, id)`` and never needs to know
  which model wrote a story.

Do not add fields casually: changing this file changes the interface between team members.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Dataset = Literal["vist", "vwp", "aesop"]
Split = Literal["train", "val", "test"]

DATASETS: tuple[str, ...] = ("vist", "vwp", "aesop")
SPLITS: tuple[str, ...] = ("train", "val", "test")

MIN_IMAGES = 3
MAX_IMAGES = 10


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class StoryRecord(_Strict):
    """One reference story: an image sequence and its texts.

    Attributes:
        id: Unique within ``(dataset, split)``, e.g. the VIST ``story_id``.
        dataset: Source dataset.
        split: Data split.
        images: Ordered image paths (3–10), relative to the repo root or absolute.
        story: SIS story text (what we train on). Placeholders already rewritten.
        dis: Optional per-image *descriptions in sequence*; empty or ``len(images)``.
        dii: Optional per-image *descriptions in isolation*; empty or ``len(images)``.
    """

    id: str = Field(min_length=1)
    dataset: Dataset
    split: Split
    images: list[str] = Field(min_length=MIN_IMAGES, max_length=MAX_IMAGES)
    story: str = Field(min_length=1)
    dis: list[str] = Field(default_factory=list)
    dii: list[str] = Field(default_factory=list)

    @field_validator("images")
    @classmethod
    def _non_empty_paths(cls, v: list[str]) -> list[str]:
        if any(not p.strip() for p in v):
            raise ValueError("image paths must be non-empty strings")
        return v

    @model_validator(mode="after")
    def _aligned_descriptions(self) -> StoryRecord:
        for name in ("dis", "dii"):
            texts = getattr(self, name)
            if texts and len(texts) != len(self.images):
                raise ValueError(
                    f"{name} has {len(texts)} entries but there are {len(self.images)} images"
                )
        return self


class GeneratedStory(_Strict):
    """One model output for one sequence.

    Attributes:
        id, dataset, split: Same key as the :class:`StoryRecord` it was generated from.
        model: HF model id, e.g. ``Qwen/Qwen3-VL-2B-Instruct``.
        adapter: ``None`` for zero-shot, otherwise a local path or HF Hub id of the LoRA adapter.
        prompt_id: Key in ``configs/prompts.yaml`` (e.g. ``P0``, ``P1``).
        story: Generated text (post-processed, no chat template remnants).
        gen_config: Decoding settings used (temperature, top_p, max_new_tokens, seed, …).
    """

    id: str = Field(min_length=1)
    dataset: Dataset
    split: Split
    model: str = Field(min_length=1)
    adapter: str | None = None
    prompt_id: str = Field(min_length=1)
    story: str
    gen_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("adapter")
    @classmethod
    def _adapter_not_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("adapter must be null or a non-empty path / hub id")
        return v

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.dataset, self.split, self.id)


RECORD_TYPES: dict[str, type[_Strict]] = {
    "story": StoryRecord,
    "generated": GeneratedStory,
}
