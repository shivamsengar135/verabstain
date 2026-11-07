"""Download the arXiv full text of every paper cited in the manuscript (keys from paper/main.aux; arXiv ids from
paper/refs_verification.csv) into literature/pdfs/<id>.pdf and extract plain text to literature/pdfs/<id>.txt, for the
Rule 7 full-text check of each statement about a cited paper. Polite: one request every 3 s, skips files already present.
Usage: .venv/Scripts/python.exe scripts/fetch_cited_fulltext.py
"""
from __future__ import annotations

import csv
import re
import time
import urllib.request
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "literature" / "pdfs"
UA = "ragverify-research-script/0.1 (citation full-text check)"


def cited() -> list[str]:
    aux = (ROOT / "paper" / "main.aux").read_text(encoding="utf-8", errors="ignore")
    return sorted({k for grp in re.findall(r"\\citation\{([^}]*)\}", aux) for k in grp.split(",")})


def main() -> None:
    meta = {r["key"]: r for r in csv.DictReader((ROOT / "paper" / "refs_verification.csv").open(encoding="utf-8"))}
    OUT.mkdir(parents=True, exist_ok=True)
    for k in cited():
        aid = meta.get(k, {}).get("arxiv_id", "").strip()
        if not aid:
            print(k, "no arXiv id (check by DOI)", flush=True)
            continue
        pdf, txt = OUT / f"{aid}.pdf", OUT / f"{aid}.txt"
        if not pdf.exists():
            req = urllib.request.Request(f"https://arxiv.org/pdf/{aid}", headers={"User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if not data.startswith(b"%PDF"):
                    print(k, aid, "not a PDF", flush=True)
                    continue
                pdf.write_bytes(data)
            except Exception as e:  # noqa: BLE001
                print(k, aid, "download failed:", type(e).__name__, e, flush=True)
                continue
            time.sleep(3)
        if not txt.exists():
            try:
                text = "\n".join(p.extract_text() or "" for p in PdfReader(pdf).pages)
                txt.write_text(text.encode("utf-8", "replace").decode("utf-8"), encoding="utf-8")  # drop lone surrogates
            except Exception as e:  # noqa: BLE001
                print(k, aid, "text extraction failed:", type(e).__name__, flush=True)
                continue
        print(k, aid, "ok", txt.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
