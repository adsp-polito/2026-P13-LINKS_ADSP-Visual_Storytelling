"""data/download_concreteness.py: reading the official Brysbaert .xlsx layout, offline."""

from __future__ import annotations

import io
import zipfile

import pytest


def _xlsx(rows):
    strings = sorted({v for r in rows for v in r if isinstance(v, str)})
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    sst = "".join(f"<si><t>{s}</t></si>" for s in strings)
    body = ""
    for i, row in enumerate(rows, start=1):
        cells = ""
        for col, v in zip("ABC", row, strict=False):
            if isinstance(v, str):
                cells += f'<c r="{col}{i}" t="s"><v>{strings.index(v)}</v></c>'
            else:
                cells += f'<c r="{col}{i}"><v>{v}</v></c>'
        body += f'<row r="{i}">{cells}</row>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/sharedStrings.xml", f"<sst {ns}>{sst}</sst>")
        z.writestr(
            "xl/worksheets/sheet1.xml", f"<worksheet {ns}><sheetData>{body}</sheetData></worksheet>"
        )
    return buf.getvalue()


def test_download_concreteness_reads_the_official_sheet_layout():
    import importlib.util

    spec = importlib.util.spec_from_file_location("dl", "data/download_concreteness.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    rows = dl.read_rows(
        _xlsx([["Word", "Bigram", "Conc.M"], ["dog", 0, 4.85], ["infinity", 0, 1.36]])
    )
    assert [(r["Word"], float(r["Conc.M"])) for r in rows] == [("dog", 4.85), ("infinity", 1.36)]
    with pytest.raises(ValueError, match="expected 39954 words"):
        dl.convert(_xlsx([["Word", "Bigram", "Conc.M"], ["dog", 0, 4.85]]))


def test_download_concreteness_checks_the_official_file_hash():
    import hashlib
    import importlib.util

    spec = importlib.util.spec_from_file_location("dl", "data/download_concreteness.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    with pytest.raises(ValueError, match="Springer changed the file"):
        dl.check_sha256(b"not the official file")
    data = b"pretend official"
    dl.EXPECTED_SHA256 = hashlib.sha256(data).hexdigest()
    dl.check_sha256(data)
