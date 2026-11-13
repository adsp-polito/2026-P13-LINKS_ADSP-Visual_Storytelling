"""Logging setup and optional Weights & Biases integration.

W&B is off unless ``logging.wandb: true`` in the config. Credentials come from ``.env``
(``WANDB_API_KEY``); nothing is hard-coded.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

if TYPE_CHECKING:
    from p13.common.config import ExperimentConfig

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once and load ``.env`` into the environment."""
    load_dotenv(override=False)
    logging.basicConfig(level=level.upper(), format=_FMT, datefmt="%H:%M:%S", force=True)


def init_wandb(cfg: ExperimentConfig, job_type: str) -> Any | None:
    """Start a W&B run if enabled in the config; return the run or ``None``."""
    if not cfg.logging.wandb:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        logging.getLogger(__name__).warning("logging.wandb is true but WANDB_API_KEY is unset")
    import wandb  # optional dependency ([train] extra)

    return wandb.init(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity or os.environ.get("WANDB_ENTITY"),
        name=f"{cfg.experiment.id}_{cfg.experiment.name}",
        job_type=job_type,
        config=cfg.to_dict(),
    )
