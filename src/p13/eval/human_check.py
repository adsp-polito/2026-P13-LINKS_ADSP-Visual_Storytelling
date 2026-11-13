"""Human check: 40 sequences × 3 raters.  **Stub – owner: Nioosha.**

Export:

* Sample ``cfg.eval.human_check_n`` ids from the dev subset (seed 42, :func:`sample_ids`).
* Blind with :func:`p13.eval.blinding.make_pairs` / ``write_blind`` – raters never see model
  names. Keep the key file out of the rater folder.
* Produce one CSV (or Google-Form-ready sheet) per rater with: item id, image file names
  (images shared privately, **never** committed), story A, story B, and columns for
  A/B/tie/both_bad + per-story hallucination flags (invented name / date / place / object).

Import:

* Read rater sheets back, unblind, compute inter-rater agreement (Fleiss' κ) and the
  majority verdict, and use it to validate the LLM judge and the automatic metrics.
"""

from __future__ import annotations

from pathlib import Path

from p13.common.config import ExperimentConfig


def export_human_check(cfg: ExperimentConfig, left: Path, right: Path, out_dir: Path) -> Path:
    raise NotImplementedError("human-check export not implemented yet (Nioosha)")


def import_human_check(cfg: ExperimentConfig, sheets_dir: Path, key_path: Path) -> Path:
    raise NotImplementedError("human-check import not implemented yet (Nioosha)")
