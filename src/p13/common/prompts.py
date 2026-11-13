"""Prompt registry backed by ``configs/prompts.yaml``."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml


@cache
def load_prompts(path: str | Path = "configs/prompts.yaml") -> dict[str, str]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    prompts = data.get("prompts", {})
    out = {}
    for pid, entry in prompts.items():
        text = entry["text"] if isinstance(entry, dict) else entry
        out[pid] = " ".join(str(text).split())
    return out


def get_prompt(prompt_id: str, path: str | Path = "configs/prompts.yaml") -> str:
    prompts = load_prompts(path)
    if prompt_id not in prompts:
        raise KeyError(f"unknown prompt id {prompt_id!r}; available: {sorted(prompts)}")
    return prompts[prompt_id]
