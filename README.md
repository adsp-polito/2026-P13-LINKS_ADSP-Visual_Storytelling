# P13 – Efficient Adaptation of Multimodal LLMs for Visual Storytelling

Applied Data Science Project (ADSP), Politecnico di Torino, 2026/27 – **Team 10**.
Proposed by **LINKS Foundation** within the EU project **NARRATIA**.

**Task.** Given a sequence of 3–10 images, generate a coherent story that is **grounded** in
the images – no invented names, dates, places or objects.

**Approach.** Take Qwen3-VL-Instruct (2B / 4B / 8B), keep it **frozen**, and adapt it with
parameter-efficient fine-tuning (LoRA / QLoRA) on human stories (SIS). Compare adapter OFF
(zero-shot) vs adapter ON (fine-tuned) on the validation splits of VIST, VWP and AESOP.

| Step | What | Where |
|---|---|---|
| 1 | Data exploration, unified JSONL | `src/p13/data/`, `notebooks/` |
| 2 | Zero-shot evaluation (E0) | `src/p13/infer/`, `configs/e0_*.yaml` |
| 3 | LoRA / QLoRA fine-tuning (E1–E7) | `src/p13/train/`, `configs/e1_*.yaml` … |
| 4 | Evaluation on validation splits | `src/p13/eval/` |

> **Status:** this repository is a **template**. Shared plumbing (schema, JSONL I/O, config,
> seeding, CLI) is implemented and tested; the project work itself is stubbed with
> `NotImplementedError` and docstrings describing what to build.

## Setup

Requires Python 3.11.

```bash
# with uv (recommended)
uv venv -p 3.11
uv pip install -e ".[dev]"            # light: schema, config, tests (CPU only)
uv pip install -e ".[dev,train]"      # + torch / transformers / peft / trl / bitsandbytes
uv pip install -e ".[dev,train,eval]" # + metric libraries

# or with pip
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pre-commit install                    # ruff + "no images / no raw data" guards on commit
cp .env.example .env                  # then fill HF_TOKEN, WANDB_API_KEY (never commit .env)
```

## Folder map

```
configs/            one YAML per experiment (E0 zero-shot, E1 LoRA, …) + prompts.yaml
data/               download/preprocess scripts + ID lists ONLY (no images, no raw files)
src/p13/data/       dataset loaders → unified JSONL         (owner: Parsa)
src/p13/infer/      zero-shot + adapter generation          (owner: Behrad)
src/p13/train/      LoRA/QLoRA SFT training                 (owner: Behrad)
src/p13/eval/       metrics, LLM judge, human-check export  (owner: Nioosha)
src/p13/common/     JSONL schema, config loading, seeding, logging (shared)
outputs/            stories JSONL, scores CSV (gitignored)
notebooks/          exploration notebooks
report/             checkpoint slides material, figures
tests/
```

## The unified JSONL contract

Every module reads and writes this format (`src/p13/common/schema.py`), so each person can
work independently and evaluation never needs to know which model wrote a story.

```jsonc
// reference story (data/processed/<dataset>_<split>.jsonl)
{"id": "45530", "dataset": "vist", "split": "val", "images": ["data/images/vist/val/1.jpg", ...],
 "story": "...", "dis": ["...", ...], "dii": ["...", ...]}

// generated story (outputs/<E-id>_<name>/stories_*.jsonl)
{"id": "45530", "dataset": "vist", "split": "val", "model": "Qwen/Qwen3-VL-2B-Instruct",
 "adapter": null, "prompt_id": "P1", "story": "...", "gen_config": {"max_new_tokens": 256}}
```

```python
from p13.common import StoryRecord, GeneratedStory, read_jsonl, write_jsonl

records = read_jsonl("data/processed/vist_val.jsonl", StoryRecord)  # validated
```

Changes to the schema need a PR reviewed by the whole team.

## Running

All commands take `--config`, accept `--set key.sub=value` overrides and `--dry-run`.

```bash
p13-prepare  --config configs/e0_zeroshot.yaml --split all            # raw → JSONL + dev-200 ids
p13-generate --config configs/e0_zeroshot.yaml                        # zero-shot (adapter OFF)
p13-train    --config configs/e1_lora_2b_6gb.yaml                     # QLoRA on a ~6 GB GPU
p13-train    --config configs/e1_lora_4b_24gb.yaml                    # LoRA on a 24 GB GPU
p13-generate --config configs/e1_lora_2b_6gb.yaml --set model.adapter=outputs/E1_lora_2b_6gb/adapter
p13-eval     --config configs/e0_zeroshot.yaml --stories outputs/E0_zeroshot_2b/stories_*.jsonl
```

**Switching GPU** – only these keys matter: `model.name` (2B/4B/8B), `model.load_in_4bit`,
`data.max_image_side` (default 448), `train.per_device_batch_size`, `train.grad_accum`.

## Data policy (GDPR / licensing)

* **Never commit images or raw dataset files.** `data/raw/`, `data/images/`, `data/processed/`,
  `outputs/**`, `*.safetensors`, `wandb/` and `.env` are gitignored; a pre-commit hook and a
  CI test fail if image files are tracked. Only `report/figures/` may contain images.
* Datasets are downloaded by each member following `data/README.md`.
* Secrets (HF token, W&B key) live only in `.env`.
* Adapters are pushed to the Hugging Face Hub as **private** repos.
* Reproducibility: seed 42 everywhere; fixed 200-sequence dev subset per dataset in
  `data/splits/`.

## Team

| Member | Area | Folders |
|---|---|---|
| Parsa | Data | `src/p13/data/`, `data/`, dataset notebooks |
| Behrad | Inference & training | `src/p13/infer/`, `src/p13/train/`, `configs/` |
| Nioosha | Evaluation | `src/p13/eval/`, `report/` |

Mentors (LINKS Foundation): Federico D'Asaro, Juan José Márquez Villacís, Giacomo Blanco.

Milestones: Checkpoint 2 – Wed 2 Dec 2026 · Checkpoint 3 – from Fri 18 Dec 2026.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow. Licensed under Apache-2.0.
