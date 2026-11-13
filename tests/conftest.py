"""Tiny fake records shared by the tests. No real images or dataset files are used."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def story_dict() -> dict:
    return {
        "id": "s1",
        "dataset": "vist",
        "split": "val",
        "images": [
            "data/images/vist/val/1.jpg",
            "data/images/vist/val/2.jpg",
            "data/images/vist/val/3.jpg",
        ],
        "story": "A family met at the beach. They ate together. The dog played in the sand.",
        "dis": [],
        "dii": [],
    }


@pytest.fixture
def generated_dict() -> dict:
    return {
        "id": "s1",
        "dataset": "vist",
        "split": "val",
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "adapter": None,
        "prompt_id": "P1",
        "story": "People gather on a beach. They share food. A dog runs on the sand.",
        "gen_config": {"max_new_tokens": 256, "do_sample": False, "seed": 42},
    }


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test from the repo root so ``configs/...`` paths resolve."""
    monkeypatch.chdir(REPO_ROOT)
