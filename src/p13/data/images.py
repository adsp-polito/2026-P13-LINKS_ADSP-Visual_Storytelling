"""Image checks and resizing shared by all dataset loaders."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff")


def find_image(directory: Path, stem: str) -> Path | None:
    """Return ``directory/<stem>.<ext>`` for the first known extension that exists."""
    for ext in IMAGE_EXTENSIONS:
        for cand in (directory / f"{stem}{ext}", directory / f"{stem}{ext.upper()}"):
            if cand.is_file():
                return cand
    return None


def is_valid_image(path: Path) -> bool:
    """True if PIL can fully decode the file (truncated / corrupt / empty files fail)."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:  # verify() leaves the image unusable; reopen and decode
            im.load()
            return im.width > 0 and im.height > 0
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
        return False


def resize_long_side(src: Path, dst: Path, max_side: int = 448, quality: int = 90) -> Path:
    """Write ``src`` as RGB JPEG to ``dst`` with its long side ≤ ``max_side`` (never upscales).

    Skips the work if ``dst`` already exists (preprocessing is resumable).
    """
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        im.save(dst, format="JPEG", quality=quality)
    return dst
