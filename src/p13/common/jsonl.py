"""Read / write / validate the unified JSONL files.

Example::

    from p13.common import StoryRecord, read_jsonl, write_jsonl

    records = read_jsonl("data/processed/vist_val.jsonl", StoryRecord)
    write_jsonl("outputs/copy.jsonl", records)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from p13.common.schema import GeneratedStory, StoryRecord

T = TypeVar("T", bound=BaseModel)


class JSONLError(ValueError):
    """A line in a JSONL file is malformed or violates the schema."""

    def __init__(self, path: Path | str, line_no: int, message: str):
        self.path = Path(path)
        self.line_no = line_no
        super().__init__(f"{path}:{line_no}: {message}")


def iter_jsonl(path: str | Path, model: type[T]) -> Iterator[T]:
    """Yield validated records one by one. Blank lines are skipped."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise JSONLError(path, line_no, f"invalid JSON: {e.msg}") from e
            try:
                yield model.model_validate(obj)
            except ValidationError as e:
                raise JSONLError(path, line_no, _short(e)) from e


def read_jsonl(path: str | Path, model: type[T], *, unique_ids: bool = True) -> list[T]:
    """Read a whole file. By default also checks ``(dataset, split, id)`` is unique."""
    records = list(iter_jsonl(path, model))
    if unique_ids:
        _check_unique(path, records)
    return records


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> int:
    """Write records (one compact JSON object per line, UTF-8). Returns the count.

    Parent directories are created. The file is written to ``<path>.tmp`` and renamed,
    so a crash never leaves a half-written file behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def detect_model(path: str | Path) -> type[StoryRecord] | type[GeneratedStory]:
    """Guess the record type from the first non-blank line (``model`` key ⇒ generated)."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return GeneratedStory if "model" in json.loads(line) else StoryRecord
    return StoryRecord


def validate_file(path: str | Path, model: type[BaseModel] | None = None) -> int:
    """Validate a file fully; return the number of records or raise :class:`JSONLError`."""
    model = model or detect_model(path)
    return len(read_jsonl(path, model))


def _check_unique(path: str | Path, records: list[T]) -> None:
    seen: set[tuple] = set()
    for i, r in enumerate(records, start=1):
        key = (getattr(r, "dataset", None), getattr(r, "split", None), getattr(r, "id", None))
        if isinstance(r, GeneratedStory):
            key = (*key, r.model, r.adapter, r.prompt_id)
        if key in seen:
            raise JSONLError(path, i, f"duplicate key {key}")
        seen.add(key)


def _short(e: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(x) for x in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors()
    )
