"""Hide which system wrote a story (the "model label hidden" step).  **Stub – owner: Nioosha.**

Used by the pairwise LLM judge and the human check: raters see "story A" / "story B" in a
random (seeded) order per item, and the key mapping back to the real
``(model, adapter, prompt_id)`` is stored in a separate file that raters never see.
"""

from __future__ import annotations

from pathlib import Path

from p13.common.schema import GeneratedStory


def make_pairs(left: list[GeneratedStory], right: list[GeneratedStory], seed: int = 42) -> list:
    """Pair stories with the same ``(dataset, split, id)`` and randomise A/B per item."""
    raise NotImplementedError("TODO (Nioosha)")


def write_blind(pairs: list, items_path: Path, key_path: Path) -> None:
    """Write rater-facing items (no system names) and the private unblinding key."""
    raise NotImplementedError("TODO (Nioosha)")


def unblind(verdicts: dict[str, str], key: dict[str, dict[str, str]]) -> dict[str, str]:
    """Map per-item verdicts ``A|B|tie|both_bad`` back to system names."""
    raise NotImplementedError("TODO (Nioosha)")
