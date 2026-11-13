"""Automatic metrics.  **Stubs – owner: Nioosha.**

Every metric takes generated stories (and, where needed, the matching references / images)
and returns per-item scores so :mod:`p13.eval.report` can bootstrap confidence intervals.

Signature convention::

    def metric(gens: list[GeneratedStory], refs: dict[key, StoryRecord]) -> dict[key, float]

where ``key = (dataset, split, id)``.

Groups (match the project diagram):

* **Headline** – GROOViST, RoViST-C, RoViST-NR, dHM, hallucination rate.
* **Supporting** – CLIPScore, distinct-2, length, reading grade.
* **Comparability** (with prior VIST work) – BLEU-4, METEOR, CIDEr, ROUGE-L, BERTScore.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

from p13.common.config import ExperimentConfig
from p13.common.schema import GeneratedStory, StoryRecord

Key: TypeAlias = tuple[str, str, str]
Scores: TypeAlias = dict[Key, float]
Refs: TypeAlias = dict[Key, StoryRecord]


def _todo(name: str) -> NotImplementedError:
    return NotImplementedError(f"metric {name!r} not implemented yet (Nioosha)")


# ---------------------------------------------------------------- headline


def groovist(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """GROOViST (Surikuchi et al., 2023): visual grounding of noun phrases to image regions
    via CLIP similarity, with temporal weighting. Reference-free."""
    raise _todo("groovist")


def rovist_c(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """RoViST coherence (Wang et al., 2022): ALBERT-based next-sentence coherence score."""
    raise _todo("rovist_c")


def rovist_nr(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """RoViST non-redundancy: penalises repeated n-grams within and across sentences."""
    raise _todo("rovist_nr")


def dhm(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """dHM: distance to human-level on (grounding, coherence, repetition), per Surikuchi et
    al. Combine GROOViST / RoViST-C / RoViST-NR against the human reference stories."""
    raise _todo("dhm")


def hallucination_rate(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """Share of invented entities: (a) named entities (spaCy NER: PERSON, GPE, DATE, ORG…)
    absent from every image description; (b) nouns/objects not detected in any image
    (open-vocabulary detector or DII/DIS text). Lower is better."""
    raise _todo("hallucination_rate")


# ---------------------------------------------------------------- supporting


def clipscore(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """Mean CLIPScore between each sentence and its aligned image (sentence i ↔ image i)."""
    raise _todo("clipscore")


def distinct_2(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """Ratio of unique bigrams to total bigrams per story (lexical diversity)."""
    raise _todo("distinct_2")


def length(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """Story length in words (P1 asks for 50–100)."""
    raise _todo("length")


def reading_grade(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """Flesch–Kincaid grade level (e.g. via ``textstat``)."""
    raise _todo("reading_grade")


# ---------------------------------------------------------------- comparability


def bleu4(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """BLEU-4 vs SIS reference(s) (pycocoevalcap, corpus + per-item)."""
    raise _todo("bleu4")


def meteor(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """METEOR vs SIS reference(s)."""
    raise _todo("meteor")


def cider(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """CIDEr-D vs SIS reference(s). Note: corpus-dependent IDF – compute on the full set."""
    raise _todo("cider")


def rouge_l(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """ROUGE-L F1 vs SIS reference(s)."""
    raise _todo("rouge_l")


def bertscore(gens: list[GeneratedStory], refs: Refs) -> Scores:
    """BERTScore F1 vs SIS reference(s) (roberta-large, rescaled with baseline)."""
    raise _todo("bertscore")


def evaluate(cfg: ExperimentConfig, story_files: list[Path]) -> Path:
    """Run every metric in ``cfg.eval.metrics`` on the given GeneratedStory files.

    Load references from ``<jsonl_dir>/<dataset>_<split>.jsonl``, call each function in
    :data:`METRICS`, write per-item scores to ``cfg.run_dir / "scores.csv"`` and return it.
    """
    raise NotImplementedError("metric pipeline not implemented yet (Nioosha)")


METRICS: dict[str, Callable[[list[GeneratedStory], Refs], Scores]] = {
    "groovist": groovist,
    "rovist_c": rovist_c,
    "rovist_nr": rovist_nr,
    "dhm": dhm,
    "hallucination_rate": hallucination_rate,
    "clipscore": clipscore,
    "distinct_2": distinct_2,
    "length": length,
    "reading_grade": reading_grade,
    "bleu4": bleu4,
    "meteor": meteor,
    "cider": cider,
    "rouge_l": rouge_l,
    "bertscore": bertscore,
}
