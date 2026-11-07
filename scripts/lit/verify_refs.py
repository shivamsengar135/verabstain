"""Verify literature entries online and build BibTeX (Rule 7).

Input : literature/seed_refs.csv  (key,title,arxiv_id,category,role)
Output: paper/refs_verification.csv  one row per key
        literature/meta/<key>.json    raw records (arXiv abstract/comment/journal_ref, S2 record)
        paper/refs.bib                BibTeX: ACL Anthology > DOI content negotiation > built from S2+arXiv

Sources (DBLP blocks scripted clients with a bot check, so it is not used):
  * arXiv API (batch by ID; title search when no ID)  -> title, authors, abstract, comment, journal_ref
  * Semantic Scholar Graph API (batch by ARXIV:id; title search otherwise) -> venue, year, DOI, ACL id
  * aclanthology.org/<id>.bib and doi.org content negotiation for BibTeX
Status: VERIFIED if the online title matches the seed title (normalised similarity >= 0.92) and an
arXiv ID (when given) resolves to that title; CHECK otherwise. Venue/year must still be eyeballed
for conference papers whose arXiv year differs from the venue year (flag column `year_note`).
Usage: .venv/Scripts/python.exe scripts/lit/verify_refs.py
"""
from __future__ import annotations

import csv
import difflib
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "literature" / "seed_refs.csv"
META = ROOT / "literature" / "meta"
OUT = ROOT / "paper" / "refs_verification.csv"
BIB = ROOT / "paper" / "refs.bib"
UA = {"User-Agent": "ragverify-refcheck/0.1 (academic literature verification)"}
CTX = ssl.create_default_context(cafile=certifi.where())
S2F = "title,venue,year,externalIds,authors,publicationVenue,publicationDate,journal"
FIELDS = ["key", "status", "title_seed", "title_found", "title_sim", "authors", "venue", "year",
          "year_note", "doi", "arxiv_id", "acl_id", "arxiv_comment", "bib_source", "checked_at", "notes"]


def http(url: str, data: bytes | None = None, headers: dict | None = None, tries: int = 6) -> bytes:
    h = dict(UA, **(headers or {}))
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=h)
            with urllib.request.urlopen(req, context=CTX, timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep((30 if e.code == 429 else 5) * (i + 1))
                continue
            raise
        except Exception:  # noqa: BLE001
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"failed after retries: {url}")


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (t or "").lower())).strip()


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def parse_arxiv(xml: bytes) -> list[dict]:
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    out = []
    for e in ET.fromstring(xml).findall("a:entry", ns):
        t = " ".join(e.findtext("a:title", "", ns).split())
        if not t or t == "Error":
            continue
        out.append({
            "title": t,
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
            "abstract": " ".join(e.findtext("a:summary", "", ns).split()),
            "published": e.findtext("a:published", "", ns)[:10],
            "arxiv_id": re.sub(r"v\d+$", "", e.findtext("a:id", "", ns).rsplit("/abs/", 1)[-1]),
            "comment": " ".join((e.findtext("x:comment", "", ns) or "").split()),
            "journal_ref": " ".join((e.findtext("x:journal_ref", "", ns) or "").split()),
        })
    return out


def arxiv_batch(ids: list[str]) -> dict[str, dict]:
    res = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        xml = http("https://export.arxiv.org/api/query?max_results=60&id_list=" + ",".join(chunk))
        for rec in parse_arxiv(xml):
            res[rec["arxiv_id"]] = rec
        time.sleep(3)
    return res


def arxiv_search(title: str) -> dict | None:
    q = urllib.parse.quote(f'ti:"{norm(title)}"')
    recs = parse_arxiv(http(f"https://export.arxiv.org/api/query?max_results=5&search_query={q}"))
    time.sleep(3)
    return max(recs, key=lambda r: sim(r["title"], title)) if recs else None


def s2_batch(arxiv_ids: list[str]) -> dict[str, dict]:
    res = {}
    for i in range(0, len(arxiv_ids), 400):
        chunk = arxiv_ids[i:i + 400]
        body = json.dumps({"ids": [f"ARXIV:{a}" for a in chunk]}).encode()
        data = json.loads(http(f"https://api.semanticscholar.org/graph/v1/paper/batch?fields={S2F}", data=body,
                               headers={"Content-Type": "application/json"}))
        for a, rec in zip(chunk, data):
            if rec:
                res[a] = rec
        time.sleep(3)
    return res


def s2_search(title: str) -> dict | None:
    q = urllib.parse.quote(norm(title))
    data = json.loads(http(f"https://api.semanticscholar.org/graph/v1/paper/search?limit=5&fields={S2F}&query={q}"))
    time.sleep(3)
    recs = data.get("data") or []
    return max(recs, key=lambda r: sim(r.get("title", ""), title)) if recs else None


def openalex_lookup(title: str) -> dict | None:
    """Best title match on OpenAlex, mapped to the S2-like dict used below; prefers non-arXiv venues."""
    if not title:
        return None
    sel = "id,title,publication_year,doi,primary_location,locations,authorships"
    data = json.loads(http("https://api.openalex.org/works?per_page=8&select=" + sel + "&search=" + urllib.parse.quote(norm(title))))
    time.sleep(1)
    best = None
    for w in data.get("results", []):
        s = sim(w.get("title") or "", title)
        if s < 0.92:
            continue
        srcs = [((loc.get("source") or {}).get("display_name") or "") for loc in [w.get("primary_location") or {}] + (w.get("locations") or [])]
        venue = next((v for v in srcs if v and "arxiv" not in v.lower() and "ssrn" not in v.lower()), "")
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        rank = (s, bool(venue), not doi.lower().startswith("10.48550"))
        if best is None or rank > best[0]:
            ext = {"DOI": doi} if doi else {}
            if doi.lower().startswith("10.18653/v1/"):
                ext["ACL"] = doi.split("v1/", 1)[1]
            best = (rank, {"title": w.get("title"), "venue": venue or "arXiv", "year": w.get("publication_year"),
                           "externalIds": ext, "authors": [{"name": a["author"]["display_name"]} for a in w.get("authorships", [])],
                           "source": "openalex", "openalex_id": w.get("id")})
    return best[1] if best else None


def bib_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%")


def build_bib(key: str, s2: dict | None, ax: dict | None) -> tuple[str, str]:
    ext = (s2 or {}).get("externalIds") or {}
    if ext.get("ACL"):
        try:
            b = http(f"https://aclanthology.org/{ext['ACL']}.bib").decode("utf-8", "replace").strip()
            return re.sub(r"^@(\w+)\{[^,]+,", lambda m: f"@{m.group(1)}{{{key},", b, count=1) + "\n", "acl-anthology"
        except Exception:  # noqa: BLE001
            pass
    doi = ext.get("DOI", "")
    if doi and not doi.lower().startswith("10.48550"):
        try:
            b = http(f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex"}).decode("utf-8", "replace").strip()
            return re.sub(r"^@(\w+)\{[^,]+,", lambda m: f"@{m.group(1)}{{{key},", b, count=1) + "\n", "doi"
        except Exception:  # noqa: BLE001
            pass
    title = (ax or {}).get("title") or (s2 or {}).get("title", "")
    authors = (ax or {}).get("authors") or [a["name"] for a in (s2 or {}).get("authors", [])]
    venue = (s2 or {}).get("venue") or ""
    year = (s2 or {}).get("year") or ((ax or {}).get("published", "")[:4])
    aid = (ax or {}).get("arxiv_id", "")
    lines = [f"  author = {{{' and '.join(authors)}}}", f"  title = {{{{{bib_escape(title)}}}}}", f"  year = {{{year}}}"]
    if venue and "arxiv" not in venue.lower():
        kind = "article" if (s2 or {}).get("journal") and "journal" in str((s2 or {}).get("publicationVenue") or {}).lower() else "inproceedings"
        lines.insert(2, f"  {'journal' if kind == 'article' else 'booktitle'} = {{{bib_escape(venue)}}}")
    else:
        kind = "misc"
    if aid:
        lines += [f"  eprint = {{{aid}}}", "  archivePrefix = {arXiv}", f"  url = {{https://arxiv.org/abs/{aid}}}"]
    return f"@{kind}{{{key},\n" + ",\n".join(lines) + "\n}\n", "built-s2-arxiv"


def main() -> None:
    META.mkdir(parents=True, exist_ok=True)
    seeds = list(csv.DictReader(SEED.open(encoding="utf-8")))
    ids = [s["arxiv_id"].strip() for s in seeds if s["arxiv_id"].strip()]
    ax_by_id = arxiv_batch(ids)
    try:
        s2_by_id = s2_batch(ids)
    except Exception as e:  # noqa: BLE001
        print("S2 batch failed, continuing with arXiv + OpenAlex only:", e, flush=True)
        s2_by_id = {}
    rows, bibs = [], []
    for s in seeds:
        key, title, aid = s["key"], s["title"].strip(), s["arxiv_id"].strip()
        notes = []
        ax = ax_by_id.get(aid) if aid else None
        if aid and not ax:
            notes.append("arXiv ID did not resolve")
        s2 = s2_by_id.get(aid) if aid else None
        if s2 is None:
            try:
                s2 = openalex_lookup(title if title != "?" else (ax or {}).get("title", ""))
                if s2:
                    notes.append("venue via OpenAlex")
            except Exception as e:  # noqa: BLE001
                notes.append(f"openalex error {e}")
        if not aid:
            try:
                ax = arxiv_search(title)
                if ax and sim(ax["title"], title) < 0.92:
                    ax = None
            except Exception as e:  # noqa: BLE001
                notes.append(f"arxiv search error {e}")
        found = [t for t in [(ax or {}).get("title"), (s2 or {}).get("title")] if t]
        best = max(found, key=lambda t: sim(t, title)) if found else ""
        sbest = sim(best, title) if best else 0.0
        status = "VERIFIED" if sbest >= 0.92 else ("CHECK" if best else "NOT_FOUND")
        if title == "?" and best:
            status = "CHECK"
            notes.append("seed title unknown; take title_found after reading the abstract")
        ext = (s2 or {}).get("externalIds") or {}
        venue = (s2 or {}).get("venue") or ("arXiv" if ax else "")
        year = str((s2 or {}).get("year") or (ax or {}).get("published", "")[:4])
        ynote = ""
        if ax and venue and "arxiv" not in venue.lower() and year == ax["published"][:4]:
            ynote = "venue year may differ from arXiv year; confirm"
        bib, bsrc = ("", "")
        if status == "VERIFIED":
            try:
                bib, bsrc = build_bib(key, s2, ax)
                bibs.append(bib)
            except Exception as e:  # noqa: BLE001
                notes.append(f"bib error {e}")
        authors = [a["name"] for a in (s2 or {}).get("authors", [])] or (ax or {}).get("authors", [])
        rows.append({
            "key": key, "status": status, "title_seed": title, "title_found": best, "title_sim": f"{sbest:.3f}",
            "authors": "; ".join(authors[:15]), "venue": venue, "year": year, "year_note": ynote,
            "doi": ext.get("DOI", ""), "arxiv_id": (ax or {}).get("arxiv_id", ext.get("ArXiv", "")),
            "acl_id": ext.get("ACL", ""), "arxiv_comment": ((ax or {}).get("comment", "") + " " + (ax or {}).get("journal_ref", "")).strip()[:200],
            "bib_source": bsrc, "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"), "notes": "; ".join(notes),
        })
        (META / f"{key}.json").write_text(json.dumps({"seed": s, "arxiv": ax, "s2": s2}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{status:9s} {key:30s} {venue[:40]:40s} {year} sim={sbest:.2f} {bsrc}", flush=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    BIB.write_text("\n".join(bibs), encoding="utf-8")
    print("done", sum(r["status"] == "VERIFIED" for r in rows), "verified of", len(rows))


if __name__ == "__main__":
    main()
