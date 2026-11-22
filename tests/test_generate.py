"""Tests for story generation (CPU-only, fake model/processor, no downloads)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from p13.common.config import ExperimentConfig, parse_config_dict
from p13.common.jsonl import read_jsonl, write_jsonl
from p13.common.schema import GeneratedStory, StoryRecord
from p13.common.subset import write_ids
from p13.infer.generate import (
    build_gen_kwargs,
    build_messages,
    clean_story_text,
    gen_config_dict,
    generate,
    load_split_records,
)


def _make_images(root: Path, names: list[str]) -> list[str]:
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for name in names:
        p = img_dir / f"{name}.jpg"
        Image.new("RGB", (64, 48), "blue").save(p)
        paths.append(p.as_posix())
    return paths


def _write_fixture_split(
    root: Path,
    *,
    story_ids: list[str] = ("s1", "s2", "s3"),
    dev_ids: list[str] | None = ("s3", "s1"),
) -> ExperimentConfig:
    records = [
        StoryRecord(
            id=sid,
            dataset="vist",
            split="val",
            images=_make_images(root, [f"{sid}_1", f"{sid}_2", f"{sid}_3"]),
            story="a man walked . the man smiled . the man waved .",
        )
        for sid in story_ids
    ]
    write_jsonl(root / "processed" / "vist_val.jsonl", records)
    if dev_ids is not None:
        write_ids(root / "splits" / "vist_val_dev2.txt", list(dev_ids))

    prompts_file = root / "prompts.yaml"
    prompts_file.write_text(
        "prompts:\n  P0:\n    text: Write a story.\n  P1:\n    text: Write a grounded story.\n",
        encoding="utf-8",
    )
    return parse_config_dict(
        {
            "experiment": {"id": "E0", "name": "test_run", "seed": 42},
            "model": {"name": "Qwen/Qwen3-VL-2B-Instruct", "load_in_4bit": False},
            "data": {
                "datasets": ["vist"],
                "split": "val",
                "jsonl_dir": str(root / "processed"),
                "image_dir": str(root / "images"),
                "splits_dir": str(root / "splits"),
                "subset": "dev200" if dev_ids is not None else "full",
                "dev_subset_size": 2,
                "max_image_side": 448,
            },
            "prompt": {"file": str(prompts_file), "ids": ["P0", "P1"]},
            "generation": {"max_new_tokens": 32, "do_sample": False, "batch_size": 1},
            "output": {"dir": str(root / "outputs")},
        }
    )


class _FakeTensor:
    """Minimal 2D tensor stand-in so _run_batch works without relying on real GPU tensors."""

    def __init__(self, rows: list[list[int]]):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def numel(self) -> int:
        return self.shape[0] * self.shape[1]

    def __iter__(self):
        return iter(self.rows)


class _FakeBatchInputs(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self):
        self.calls: list[tuple[list[str], int]] = []

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True) -> str:
        assert not tokenize and add_generation_prompt
        content = messages[0]["content"]
        n_img = sum(1 for item in content if item["type"] == "image")
        text = next(item["text"] for item in content if item["type"] == "text")
        return f"<imgs={n_img}>{text}"

    def __call__(self, *, text, images, padding=True, return_tensors="pt"):
        self.calls.append((list(text), len(images)))
        return _FakeBatchInputs({"input_ids": _FakeTensor([[10, 11, 12] for _ in text])})

    def batch_decode(self, token_batches, *, skip_special_tokens=True, **kwargs) -> list[str]:
        return [
            f"<think>internal plan</think>  Generated story #{toks[-1]} .  "
            for toks in token_batches
        ]


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.step = 0

    def generate(self, **kwargs):
        input_ids: _FakeTensor = kwargs["input_ids"]
        out = []
        for row in input_ids.rows:
            self.step += 1
            out.append([*row, 100 + self.step])
        return out


def test_load_split_records_filters_dev_subset_in_order(tmp_path):
    cfg = _write_fixture_split(tmp_path, story_ids=["s1", "s2", "s3"], dev_ids=["s3", "s1"])
    records = load_split_records(cfg, "vist")
    assert [r.id for r in records] == ["s3", "s1"]


def test_load_split_records_raises_if_dev_id_missing(tmp_path):
    cfg = _write_fixture_split(tmp_path, story_ids=["s1", "s2"], dev_ids=["s1", "s999"])
    with pytest.raises(RuntimeError, match="s999"):
        load_split_records(cfg, "vist")


def test_build_messages_supports_variable_image_counts():
    for n in (3, 5, 10):
        paths = [f"data/images/p{i}.jpg" for i in range(n)]
        msgs = build_messages(paths, "Tell a story.")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert [c["type"] for c in content] == ["image"] * n + ["text"]
        assert content[-1]["text"] == "Tell a story."


def test_clean_story_text_strips_think_blocks_and_whitespace():
    raw = "  <think>step 1\nstep 2</think>\nA family   visited the park .\n\nThey had fun . "
    assert clean_story_text(raw) == "A family visited the park . They had fun ."
    # Truncated unclosed <think> block is also removed cleanly.
    assert clean_story_text("Story first . <think>cut off") == "Story first ."


def test_build_gen_kwargs_respects_do_sample(tmp_path):
    cfg_greedy = _write_fixture_split(tmp_path)
    kw_greedy = build_gen_kwargs(cfg_greedy)
    assert kw_greedy == {"max_new_tokens": 32, "do_sample": False, "repetition_penalty": 1.05}
    assert "temperature" not in kw_greedy and "top_p" not in kw_greedy

    cfg_sample = parse_config_dict({"generation": {"do_sample": True, "temperature": 0.6}})
    kw_sample = build_gen_kwargs(cfg_sample)
    assert kw_sample["do_sample"] is True
    assert kw_sample["temperature"] == 0.6
    assert "top_p" in kw_sample
    assert gen_config_dict(cfg_greedy)["seed"] == 42


def test_generate_writes_validated_stories_for_all_prompts(tmp_path):
    cfg = _write_fixture_split(tmp_path, story_ids=["s1", "s2", "s3"], dev_ids=["s3", "s1"])
    fake_model, fake_proc = _FakeModel(), _FakeProcessor()
    load_calls = 0

    def fake_loader(_cfg):
        nonlocal load_calls
        load_calls += 1
        return fake_model, fake_proc

    written = generate(cfg, loader=fake_loader, save_every=1)

    assert load_calls == 1
    assert len(written) == 2
    p0_rows = read_jsonl(written[0], GeneratedStory)
    p1_rows = read_jsonl(written[1], GeneratedStory)
    assert [r.id for r in p0_rows] == ["s3", "s1"]
    assert [r.id for r in p1_rows] == ["s3", "s1"]
    assert [r.prompt_id for r in p0_rows] == ["P0", "P0"]
    assert [r.prompt_id for r in p1_rows] == ["P1", "P1"]
    assert p0_rows[0].story == "Generated story #101 ."
    assert "<think>" not in p0_rows[0].story
    assert p0_rows[0].gen_config["seed"] == 42


def test_generate_resumes_and_skips_completed_files_without_loading_model(tmp_path):
    cfg = _write_fixture_split(tmp_path, story_ids=["s1", "s2"], dev_ids=["s1", "s2"])
    p0_path = cfg.run_dir / "stories_vist_val_P0.jsonl"
    pre_existing = GeneratedStory(
        id="s1",
        dataset="vist",
        split="val",
        model=cfg.model.name,
        adapter=None,
        prompt_id="P0",
        story="Already saved story .",
        gen_config=gen_config_dict(cfg),
    )
    write_jsonl(p0_path, [pre_existing])

    fake_model, fake_proc = _FakeModel(), _FakeProcessor()
    generate(cfg, loader=lambda _c: (fake_model, fake_proc), save_every=2)

    p0_rows = read_jsonl(p0_path, GeneratedStory)
    assert [r.id for r in p0_rows] == ["s1", "s2"]
    assert p0_rows[0].story == "Already saved story ."
    assert p0_rows[1].story == "Generated story #101 ."

    # Second run when everything is already complete must NOT even call loader().
    def fail_if_called(_c):
        raise AssertionError("loader should not be called when all stories already exist")

    generate(cfg, loader=fail_if_called)
