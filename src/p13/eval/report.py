"""Zero-shot vs fine-tuned report with 95 % bootstrap CIs.  **Stub – owner: Nioosha.**

* Input: per-item metric scores (CSV written by ``p13-eval``) for two or more systems, plus
  judge / human-check verdicts.
* For each metric: mean per system and **paired** bootstrap over items
  (``cfg.eval.bootstrap_samples`` resamples, seed 42) for the difference fine-tuned − zero-shot,
  reported as mean and ``cfg.eval.ci_level`` percentile interval.
* Judge / human: win / tie / loss / both-bad rates with bootstrap CIs.
* Output: ``scores_summary.csv`` in ``cfg.run_dir`` and figures in ``report/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from p13.common.config import ExperimentConfig


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n: int = 1000, level: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Return ``(mean_diff, low, high)`` for ``mean(b - a)`` by resampling items."""
    raise NotImplementedError("bootstrap CI not implemented yet (Nioosha)")


def build_report(cfg: ExperimentConfig, score_files: list[Path]) -> Path:
    raise NotImplementedError("report not implemented yet (Nioosha)")
