"""Pairwise multimodal LLM judge.  **Stub – owner: Nioosha.**

Design:

* Judge model: ``cfg.eval.judge_model`` (default Qwen3-VL-32B-Instruct), sees **the images**
  plus two blinded stories (see :mod:`p13.eval.blinding`).
* Verdict per item: one of ``A`` / ``B`` / ``tie`` / ``both_bad``, with a short rationale.
  Criteria: grounding (nothing invented), coherence, fit to the image order.
* **Order swap** (``cfg.eval.judge_order_swap``): judge every pair twice with A/B swapped.
  Keep the verdict only if both runs agree; otherwise record ``tie`` (log the disagreement
  rate – it measures position bias).
* **Validation**: compare judge verdicts to the 40-sequence human check (agreement, Cohen's κ)
  before trusting it on the full set.
* Output: JSONL of ``{item_id, verdict, verdict_swapped, final, rationale}`` in ``cfg.run_dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from p13.common.config import ExperimentConfig

Verdict = Literal["A", "B", "tie", "both_bad"]


def judge_pairs(cfg: ExperimentConfig, left: Path, right: Path) -> Path:
    """Judge two GeneratedStory JSONL files against each other; return the verdicts file."""
    raise NotImplementedError("LLM judge not implemented yet – see module docstring (Nioosha)")


def combine_swapped(first: Verdict, swapped: Verdict) -> Verdict:
    """Merge a verdict with its order-swapped twin (``swapped`` is in the swapped frame)."""
    raise NotImplementedError("LLM judge not implemented yet – see module docstring (Nioosha)")
