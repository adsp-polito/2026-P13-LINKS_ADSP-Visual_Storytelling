"""Download VWP v2.1 (CSV + shot images) into data/raw/vwp/.  **Owner: Parsa.**

Run from the repo root with the venv active::

    python data/download_vwp.py --split val

Safe to stop and run again: a valid image that is already on disk is skipped, and
every file is written to a temp name first, so a crash never leaves a half file.
VWP images are movie frames: academic use only, never commit them (data/README.md).
"""

from __future__ import annotations

import argparse
import http.client
import os
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from p13.data.images import is_valid_image
from p13.data.vwp import csv_name, image_urls, read_csv

HF_URL = "https://huggingface.co/datasets/tonyhong/vwp/resolve/main/{name}"


# Network errors worth a retry. URLError, HTTPError, timeouts and dropped connections are
# OSError; a connection that drops mid-download raises http.client.IncompleteRead, which is
# an HTTPException, not an OSError.
NETWORK_ERRORS = (OSError, http.client.HTTPException)


def fetch(url: str, dst: Path, retries: int = 3, timeout: int = 60) -> None:
    """Download ``url`` to ``dst`` via a temp file; retry a few times on network errors.

    If every try fails, the temp ``.part`` file is removed and the last error is raised.
    """
    tmp = dst.with_name(dst.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                tmp.write_bytes(response.read())
            os.replace(tmp, dst)
            return
        except NETWORK_ERRORS:
            if attempt == retries:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(2 * attempt)


def _download_image(job: tuple[str, str, Path]) -> str:
    """Download one image unless a valid copy is already there.

    Returns "here" (skipped), "ok" (downloaded) or "failed". Runs inside the thread pool,
    so checking the existing files is parallel too. A downloaded file that is not a valid
    image (e.g. an HTML error page) is deleted, so it is retried on the next run.
    """
    key, url, dst = job
    if dst.is_file() and is_valid_image(dst):
        return "here"
    try:
        fetch(url, dst)
    except NETWORK_ERRORS as e:
        print(f"  failed {key}: {e!r}", file=sys.stderr)
        return "failed"
    if not is_valid_image(dst):
        dst.unlink(missing_ok=True)
        print(f"  failed {key}: downloaded file is not a valid image", file=sys.stderr)
        return "failed"
    return "ok"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/vwp"))
    p.add_argument("--workers", type=int, default=8, help="parallel downloads")
    args = p.parse_args(argv)

    csv_path = args.raw_dir / csv_name(args.split)
    if not csv_path.is_file():
        print(f"downloading {csv_path.name} ...")
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        fetch(HF_URL.format(name=csv_path.name), csv_path)

    urls = image_urls(read_csv(csv_path))
    image_dir = args.raw_dir / "images" / args.split
    image_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(key, url, image_dir / f"{key}.jpg") for key, url in sorted(urls.items())]
    print(f"{args.split}: {len(jobs)} images")

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        status = Counter(pool.map(_download_image, jobs))
    failed = status["failed"]
    print(
        f"done in {time.monotonic() - start:.0f} s: {status['here']} already here, "
        f"{status['ok']} downloaded, {failed} failed"
    )
    if failed:
        print("run the script again to retry the failed images", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
