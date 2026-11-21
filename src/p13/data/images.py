"""Image checks and resizing shared by all dataset loaders."""

from __future__ import annotations

import logging
import os
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
    """True if PIL can fully decode the file.

    Truncated, corrupt, empty and too-large files (PIL's decompression-bomb limit) fail.
    """
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:  # verify() leaves the image unusable; reopen and decode
            im.load()
            return im.width > 0 and im.height > 0
    except Image.DecompressionBombError:
        log.warning("%s is too large to decode safely; treating it as broken", path)
        return False
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
        return False


def _is_up_to_date(src: Path, dst: Path, max_side: int) -> bool:
    """True if ``dst`` decodes fully and has the long side a fresh resize of ``src`` gives."""
    if not is_valid_image(dst):
        return False
    with Image.open(src) as im_src, Image.open(dst) as im_dst:
        return max(im_dst.size) == min(max_side, max(im_src.size))


def resize_long_side(src: Path, dst: Path, max_side: int = 448, quality: int = 90) -> Path:
    """Write ``src`` as RGB JPEG to ``dst`` with its long side ≤ ``max_side`` (never upscales).

    An existing ``dst`` is reused only if it decodes fully and its long side equals
    ``min(max_side, long side of src)``; otherwise it is redone. The image is written to a
    temp file and then moved into place, so a crash never leaves a half-written ``dst``.
    """
    if dst.exists() and _is_up_to_date(src, dst, max_side):
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = max_side / max(w, h)
            if scale < 1:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            im.save(tmp, format="JPEG", quality=quality)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst
