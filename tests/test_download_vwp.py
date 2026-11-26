"""Tests for data/download_vwp.py (fake network only; CI never downloads)."""

import http.client
import importlib.util
import io
from pathlib import Path

import pytest
from PIL import Image

_SCRIPT = Path(__file__).resolve().parents[1] / "data" / "download_vwp.py"
_spec = importlib.util.spec_from_file_location("download_vwp", _SCRIPT)
download_vwp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download_vwp)


def _fake_urlopen(results):
    """urlopen that raises or returns the next item of ``results`` on each call."""
    calls = iter(results)

    def urlopen(url, timeout):
        item = next(calls)
        if isinstance(item, BaseException):
            raise item
        return io.BytesIO(item)

    return urlopen


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(download_vwp.time, "sleep", lambda s: None)


def test_fetch_retries_a_dropped_connection(tmp_path, monkeypatch):
    dropped = http.client.IncompleteRead(b"half")  # an HTTPException, not an OSError
    monkeypatch.setattr(
        download_vwp.urllib.request, "urlopen", _fake_urlopen([dropped, b"image bytes"])
    )
    dst = tmp_path / "a.jpg"

    download_vwp.fetch("https://example.org/a.jpg", dst)

    assert dst.read_bytes() == b"image bytes"
    assert not (tmp_path / "a.jpg.part").exists()


def test_fetch_removes_part_file_after_last_failure(tmp_path, monkeypatch):
    errors = [OSError("timeout"), http.client.IncompleteRead(b""), OSError("reset")]
    monkeypatch.setattr(download_vwp.urllib.request, "urlopen", _fake_urlopen(errors))
    dst = tmp_path / "a.jpg"
    (tmp_path / "a.jpg.part").write_bytes(b"left from a crash")

    with pytest.raises(OSError, match="reset"):
        download_vwp.fetch("https://example.org/a.jpg", dst, retries=3)

    assert not dst.exists()
    assert not (tmp_path / "a.jpg.part").exists()


def test_download_image_skips_a_valid_existing_file(tmp_path, monkeypatch):
    dst = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8), "red").save(dst)
    monkeypatch.setattr(download_vwp.urllib.request, "urlopen", _fake_urlopen([]))  # no call

    assert download_vwp._download_image(("a", "https://example.org/a.jpg", dst)) == "here"


def test_download_image_deletes_a_file_that_is_not_an_image(tmp_path, monkeypatch):
    page = b"<html>error</html>"  # e.g. a server error page sent with HTTP 200
    monkeypatch.setattr(download_vwp.urllib.request, "urlopen", _fake_urlopen([page]))
    dst = tmp_path / "a.jpg"

    assert download_vwp._download_image(("a", "https://example.org/a.jpg", dst)) == "failed"
    assert not dst.exists()


def test_download_image_reports_ok(tmp_path, monkeypatch):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, format="JPEG")
    monkeypatch.setattr(download_vwp.urllib.request, "urlopen", _fake_urlopen([buf.getvalue()]))
    dst = tmp_path / "a.jpg"

    assert download_vwp._download_image(("a", "https://example.org/a.jpg", dst)) == "ok"
    assert dst.is_file()
