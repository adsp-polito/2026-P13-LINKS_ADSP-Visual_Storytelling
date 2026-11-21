"""Tests for image checks and resizing (p13.data.images)."""

import logging
from pathlib import Path

import pytest
from PIL import Image

from p13.data import images
from p13.data.images import is_valid_image, resize_long_side


def _make(path: Path, size=(800, 400)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "red").save(path)
    return path


def test_too_large_image_is_invalid_and_logged(tmp_path, monkeypatch, caplog):
    src = _make(tmp_path / "big.jpg")  # 320k pixels
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # more than 2x the limit -> bomb
    with caplog.at_level(logging.WARNING):
        assert not is_valid_image(src)
    assert "too large" in caplog.text


def test_resize_redoes_half_written_image(tmp_path):
    src = _make(tmp_path / "src.jpg")
    good = _make(tmp_path / "good.jpg", (448, 224)).read_bytes()
    dst = tmp_path / "out" / "src.jpg"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(good[: len(good) // 2])  # truncated, like a crash during save

    resize_long_side(src, dst, max_side=448)

    assert is_valid_image(dst)
    with Image.open(dst) as im:
        assert im.size == (448, 224)


def test_resize_redoes_image_made_with_old_max_side(tmp_path):
    src = _make(tmp_path / "src.jpg")
    dst = tmp_path / "out" / "src.jpg"
    resize_long_side(src, dst, max_side=224)

    resize_long_side(src, dst, max_side=448)

    with Image.open(dst) as im:
        assert im.size == (448, 224)


def test_resize_reuses_up_to_date_image(tmp_path):
    src = _make(tmp_path / "src.jpg")
    dst = resize_long_side(src, tmp_path / "out" / "src.jpg", max_side=448)
    before = dst.stat().st_mtime_ns

    resize_long_side(src, dst, max_side=448)

    assert dst.stat().st_mtime_ns == before  # not rewritten


def test_resize_leaves_no_partial_file_on_failure(tmp_path, monkeypatch):
    src = _make(tmp_path / "src.jpg")
    dst = tmp_path / "out" / "src.jpg"

    def crash(*args, **kwargs):
        raise OSError("simulated crash")

    monkeypatch.setattr(images.os, "replace", crash)
    with pytest.raises(OSError, match="simulated crash"):
        resize_long_side(src, dst, max_side=448)

    assert list((tmp_path / "out").iterdir()) == []  # no dst and no temp file left
