"""Build literature/matrix.csv from seed_refs.csv + refs_verification.csv + literature/annotations.csv.

annotations.csv (hand-written, one row per key): what_it_does, components (gate|hybrid|rerank|grounded|claimver|
revise|abstain|conformal|selfverify|benchmark), code_url, fits_8gb, notes. Each 'what_it_does' was written
after reading the arXiv abstract stored in literature/meta/<key>.json ('checked_against' column).
Usage: .venv/Scripts/python.exe scripts/lit/build_matrix.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
seeds = {r["key"]: r for r in csv.DictReader((ROOT / "literature" / "seed_refs.csv").open(encoding="utf-8"))}
ver = {r["key"]: r for r in csv.DictReader((ROOT / "paper" / "refs_verification.csv").open(encoding="utf-8"))}
ann_p = ROOT / "literature" / "annotations.csv"
ann = {r["key"]: r for r in csv.DictReader(ann_p.open(encoding="utf-8"))} if ann_p.exists() else {}
cols = ["key", "title", "venue_verified", "year_s2", "arxiv_id", "doi", "category", "role", "status", "what_it_does",
        "components", "code_url", "fits_8gb", "checked_against", "notes"]
with (ROOT / "literature" / "matrix.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for k, s in seeds.items():
        v, a = ver.get(k, {}), ann.get(k, {})
        w.writerow({"key": k, "title": v.get("title_found") or s["title"], "venue_verified": v.get("venue", ""),
                    "year_s2": v.get("year", ""), "arxiv_id": v.get("arxiv_id", ""), "doi": v.get("doi", ""),
                    "category": s["category"], "role": s["role"], "status": v.get("status", "UNCHECKED"),
                    "what_it_does": a.get("what_it_does", ""), "components": a.get("components", ""),
                    "code_url": a.get("code_url", ""), "fits_8gb": a.get("fits_8gb", ""),
                    "checked_against": "arXiv abstract" if a.get("what_it_does") else "", "notes": a.get("notes", "")})
print("matrix rows:", len(seeds), "annotated:", len(ann))
