"""CI guard: no dataset images, raw data, outputs, weights or secrets may be tracked by git."""

import re
import shutil
import subprocess

import pytest

FORBIDDEN = re.compile(
    r"(?i)(\.(jpe?g|png|gif|webp|bmp|tiff?|safetensors|ckpt|pt)$"
    r"|^data/(raw|images|processed)/"
    r"|^outputs/(?!\.gitkeep$)"
    r"|(^|/)\.env$)"
)
ALLOWED = re.compile(r"^report/figures/")


def _tracked_files() -> list[str]:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    res = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    if res.returncode != 0:
        pytest.skip("not a git repository")
    return res.stdout.splitlines()


def test_no_forbidden_files_tracked():
    bad = [f for f in _tracked_files() if FORBIDDEN.search(f) and not ALLOWED.match(f)]
    assert not bad, f"these files must not be committed (GDPR/licensing/secrets): {bad}"
