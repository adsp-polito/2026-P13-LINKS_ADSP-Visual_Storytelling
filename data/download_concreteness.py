"""Download the Brysbaert et al. (2014) concreteness ratings used by GROOViST.

Source: supplementary file of Brysbaert, Warriner & Kuperman (2014), "Concreteness ratings for
40 thousand generally known English word lemmas", Behavior Research Methods 46, 904–911.
The official .xlsx is converted to a tab-separated file with the columns ``Word`` and ``Conc.M``.

    python data/download_concreteness.py [--out data/raw/concreteness/brysbaert2014.tsv]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree

URL = (
    "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0403-5/"
    "MediaObjects/13428_2013_403_MOESM1_ESM.xlsx"
)
DEFAULT_OUT = Path("data/raw/concreteness/brysbaert2014.tsv")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
EXPECTED_WORDS = 39954
EXPECTED_SHA256 = "1673ead761e28833a40e82c0d20f10782955ced9366d600eafeefee0f2254545"


def check_sha256(data: bytes) -> None:
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(
            f"the downloaded file has SHA-256 {digest}, expected {EXPECTED_SHA256}: "
            "Springer changed the file, check it before updating EXPECTED_SHA256"
        )


def read_rows(xlsx: bytes) -> list[dict[str, str]]:
    with zipfile.ZipFile(io.BytesIO(xlsx)) as z:
        shared = [
            "".join(t.text or "" for t in si.iter(f"{NS}t"))
            for si in ElementTree.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{NS}si")
        ]
        sheet = ElementTree.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter(f"{NS}row"):
        cells = {}
        for c in row.iter(f"{NS}c"):
            value = c.find(f"{NS}v")
            if value is None or value.text is None:
                continue
            column = re.match(r"[A-Z]+", c.get("r", "")).group()
            cells[column] = shared[int(value.text)] if c.get("t") == "s" else value.text
        rows.append(cells)
    header = rows[0]
    return [{header[col]: val for col, val in r.items() if col in header} for r in rows[1:]]


def convert(xlsx: bytes) -> list[tuple[str, float]]:
    ratings = [(r["Word"].strip(), float(r["Conc.M"])) for r in read_rows(xlsx) if r.get("Word")]
    if len(ratings) != EXPECTED_WORDS:
        raise ValueError(f"expected {EXPECTED_WORDS} words, got {len(ratings)}: wrong file?")
    return ratings


def write_tsv(ratings: list[tuple[str, float]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["Word", "Conc.M"])
        writer.writerows(ratings)
    tmp.replace(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    check_sha256(data)
    ratings = convert(data)
    write_tsv(ratings, args.out)
    print(f"wrote {len(ratings)} ratings to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
