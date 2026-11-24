"""Command-line entry points (installed by ``pyproject.toml``).

    p13-prepare  --config CFG [--split val|train|test|all]   raw datasets → unified JSONL
    p13-generate --config CFG                                stories (adapter OFF or ON)
    p13-train    --config CFG                                LoRA / QLoRA adapter
    p13-eval     --config CFG --stories FILE [FILE ...] [--with-human]   metrics → scores CSV

Common flags: ``--set key.sub=value`` (repeatable) to override the config,
``--dry-run`` to print the resolved config and exit.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from p13.common.config import ExperimentConfig, load_config
from p13.common.logging import setup_logging
from p13.common.seed import set_seed

log = logging.getLogger("p13")


def _parser(prog: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument("--config", required=True, type=Path, help="experiment YAML in configs/")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set model.load_in_4bit=true (repeatable)",
    )
    p.add_argument("--dry-run", action="store_true", help="print resolved config and exit")
    return p


def _run(
    args: argparse.Namespace, job: str, fn: Callable[[ExperimentConfig, argparse.Namespace], object]
) -> int:
    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg.logging.level)
    if args.dry_run:
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0
    set_seed(cfg.experiment.seed)
    log.info(
        "%s: %s_%s (seed %d)", job, cfg.experiment.id, cfg.experiment.name, cfg.experiment.seed
    )
    try:
        fn(cfg, args)
    except NotImplementedError as e:
        log.error("not implemented yet: %s", e)
        return 2
    return 0


# ---------------------------------------------------------------- p13-prepare


def _prepare(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    from p13.data import aesop, vist, vwp

    loaders = {"vist": vist.prepare, "vwp": vwp.prepare, "aesop": aesop.prepare}
    splits = ["train", "val", "test"] if args.split == "all" else [args.split or cfg.data.split]
    for ds in cfg.data.datasets:
        for split in splits:
            out = loaders[ds](cfg, split)
            log.info("wrote %s", out)


def prepare_main(argv: list[str] | None = None) -> int:
    p = _parser("p13-prepare", "Preprocess raw datasets into the unified JSONL format.")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default=None)
    return _run(p.parse_args(argv), "prepare", _prepare)


# ---------------------------------------------------------------- p13-generate


def _generate(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    from p13.infer.generate import generate

    generate(cfg)


def generate_main(argv: list[str] | None = None) -> int:
    p = _parser("p13-generate", "Generate stories (zero-shot or with a LoRA adapter).")
    return _run(p.parse_args(argv), "generate", _generate)


# ---------------------------------------------------------------- p13-train


def _train(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    from p13.train.sft import train

    train(cfg)


def train_main(argv: list[str] | None = None) -> int:
    p = _parser("p13-train", "Train a LoRA / QLoRA adapter on SIS stories.")
    return _run(p.parse_args(argv), "train", _train)


# ---------------------------------------------------------------- p13-eval


def _eval(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    from p13.eval.metrics import evaluate

    evaluate(cfg, args.stories, with_human=args.with_human)


def eval_main(argv: list[str] | None = None) -> int:
    p = _parser("p13-eval", "Score generated stories (GeneratedStory JSONL).")
    p.add_argument(
        "--stories",
        nargs="+",
        type=Path,
        required=True,
        help="every story file of the experiment in one call: each run replaces scores.csv",
    )
    p.add_argument(
        "--with-human",
        action="store_true",
        help='also score the reference stories as the human baseline (model="human")',
    )
    return _run(p.parse_args(argv), "eval", _eval)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(prepare_main())
