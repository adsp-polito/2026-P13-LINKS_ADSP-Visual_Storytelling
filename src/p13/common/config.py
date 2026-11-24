"""Experiment configs: one YAML file per experiment → typed, nested dataclasses.

GPU-dependent knobs live in a handful of keys so switching hardware is a one-line change::

    model.name            Qwen/Qwen3-VL-2B-Instruct | -4B- | -8B-
    model.load_in_4bit    true (QLoRA, ~6 GB GPU) | false (bf16 LoRA, 24 GB GPU)
    data.max_image_side   448 by default (long side, pixels)
    train.per_device_batch_size / train.grad_accum

Any key can also be overridden on the command line::

    p13-train --config configs/e1_lora_2b_6gb.yaml --set model.name=Qwen/Qwen3-VL-4B-Instruct \
              --set train.grad_accum=8
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

DEFAULT_SEED = 42


@dataclass
class ExperimentSection:
    id: str = "E0"
    name: str = "unnamed"
    seed: int = DEFAULT_SEED
    notes: str = ""


@dataclass
class ModelSection:
    name: str = "Qwen/Qwen3-VL-2B-Instruct"
    load_in_4bit: bool = False
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: str = "sdpa"
    adapter: str | None = None  # path or HF Hub id of a LoRA adapter (inference only)


@dataclass
class DataSection:
    datasets: list[str] = field(default_factory=lambda: ["vist"])
    split: Literal["train", "val", "test"] = "val"
    jsonl_dir: str = "data/processed"
    raw_dir: str = "data/raw"
    image_dir: str = "data/images"
    splits_dir: str = "data/splits"
    max_image_side: int = 448
    subset: Literal["dev200", "full"] = "dev200"
    dev_subset_size: int = 200


@dataclass
class PromptSection:
    file: str = "configs/prompts.yaml"
    ids: list[str] = field(default_factory=lambda: ["P0", "P1"])
    train_id: str = "P1"


@dataclass
class GenerationSection:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    batch_size: int = 1
    backend: Literal["transformers", "vllm"] = "transformers"


@dataclass
class LoraSection:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    train_vision_tower: bool = False


@dataclass
class TrainSection:
    per_device_batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 2e-4
    epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    gradient_checkpointing: bool = True
    max_seq_len: int = 2048
    eval_steps: int = 200
    save_steps: int = 200
    train_split: str = "train"
    eval_split: str = "val"


@dataclass
class EvalSection:
    metrics: list[str] = field(
        default_factory=lambda: [
            # headline
            "groovist",
            "rovist_c",
            "rovist_nr",
            "dhm_g",
            "dhm_c",
            "dhm_nr",
            "dhm",
            "halluc_any",
            "halluc_ner",
            "halluc_obj_share",
            "n_objects",
            # supporting
            "clipscore",
            "distinct_2",
            "length",
            "reading_grade",
            # comparability
            "bleu4",
            "meteor",
            "cider",
            "rouge_l",
            "bertscore",
        ]
    )
    judge_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    judge_order_swap: bool = True
    human_check_n: int = 40
    human_check_raters: int = 3
    bootstrap_samples: int = 1000
    ci_level: float = 0.95
    allow_test: bool = False


@dataclass
class LoggingSection:
    level: str = "INFO"
    wandb: bool = False
    wandb_project: str = "p13-visual-storytelling"
    wandb_entity: str | None = None


@dataclass
class HubSection:
    push_adapter: bool = False
    repo_id: str | None = None
    private: bool = True


@dataclass
class OutputSection:
    dir: str = "outputs"


@dataclass
class ExperimentConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    model: ModelSection = field(default_factory=ModelSection)
    data: DataSection = field(default_factory=DataSection)
    prompt: PromptSection = field(default_factory=PromptSection)
    generation: GenerationSection = field(default_factory=GenerationSection)
    lora: LoraSection = field(default_factory=LoraSection)
    train: TrainSection = field(default_factory=TrainSection)
    eval: EvalSection = field(default_factory=EvalSection)
    logging: LoggingSection = field(default_factory=LoggingSection)
    hub: HubSection = field(default_factory=HubSection)
    output: OutputSection = field(default_factory=OutputSection)

    @property
    def run_dir(self) -> Path:
        """``outputs/<E-id>_<name>``: where this experiment writes stories, scores, adapters."""
        return Path(self.output.dir) / f"{self.experiment.id}_{self.experiment.name}"

    @property
    def effective_batch_size(self) -> int:
        return self.train.per_device_batch_size * self.train.grad_accum

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ConfigError(ValueError):
    pass


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load a YAML config, apply ``a.b=c`` overrides, return a validated dataclass."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    for ov in overrides or []:
        _apply_override(raw, ov)
    cfg = _build(ExperimentConfig, raw, where=str(path))
    _sanity_check(cfg)
    return cfg


def parse_config_dict(raw: dict[str, Any]) -> ExperimentConfig:
    cfg = _build(ExperimentConfig, raw, where="<dict>")
    _sanity_check(cfg)
    return cfg


def _apply_override(raw: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"override must look like key.sub=value, got {override!r}")
    key, value = override.split("=", 1)
    parts = key.strip().split(".")
    node = raw
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ConfigError(f"cannot override {key}: {p} is not a section")
    node[parts[-1]] = yaml.safe_load(value)


def _build(cls: type, data: Any, where: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping for {cls.__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)} in {cls.__name__}")
    kwargs = {}
    for name, value in data.items():
        tp = hints[name]
        sub = f"{where}:{name}"
        kwargs[name] = (
            _build(tp, value, sub) if dataclasses.is_dataclass(tp) else _coerce(tp, value, sub)
        )
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{where}: {value!r} not one of {list(args)}")
        return value
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(inner[0], value, where)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        return [_coerce(args[0], v, where) for v in value] if args else value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is float and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if tp is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if tp is str and isinstance(value, str):
        return value
    if tp is Any:
        return value
    raise ConfigError(f"{where}: expected {getattr(tp, '__name__', tp)}, got {value!r}")


def _sanity_check(cfg: ExperimentConfig) -> None:
    from p13.common.schema import DATASETS

    bad = [d for d in cfg.data.datasets if d not in DATASETS]
    if bad:
        raise ConfigError(f"data.datasets: unknown dataset(s) {bad}; expected {list(DATASETS)}")
    if cfg.data.max_image_side <= 0:
        raise ConfigError("data.max_image_side must be > 0")
    if cfg.train.per_device_batch_size < 1 or cfg.train.grad_accum < 1:
        raise ConfigError("train.per_device_batch_size and train.grad_accum must be >= 1")
    if cfg.hub.push_adapter and not cfg.hub.repo_id:
        raise ConfigError("hub.push_adapter is true but hub.repo_id is empty")
