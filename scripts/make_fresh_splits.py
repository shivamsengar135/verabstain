"""Fresh test items for the follow-up study v2 (PROTOCOL §17, 2026-11-07 13:50 entry, V2).
Writes data/splits/<ds>_fresh.ids and data/processed/<ds>_fresh.jsonl from the fixed positions registered there:
PopQA-LT test 1000-1398, TriviaQA test 1000-1499, HotpotQA test 1000-1499, PubHealth dev 300-499.
Checks that no fresh id appears in any other split file used by a run.
Usage: .venv/Scripts/python.exe scripts/make_fresh_splits.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S, P = ROOT / "data" / "splits", ROOT / "data" / "processed"
SPEC = {"popqa_lt": ("test", 1000, 1399), "triviaqa": ("test", 1000, 1500), "hotpotqa": ("test", 1000, 1500),
        "pubhealth": ("dev", 300, 500)}
for ds, (src, a, b) in SPEC.items():
    order = [l for l in (S / f"{ds}_{src}.ids").read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = order[a:b]
    assert len(ids) == b - a, (ds, len(ids))
    used = set(order[:a])
    for f in S.glob(f"{ds}_*.ids"):
        if f.name not in (f"{ds}_{src}.ids", f"{ds}_fresh.ids"):
            used |= {l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert not used & set(ids), (ds, len(used & set(ids)))
    keep = set(ids)
    lines = (P / f"{ds}_{src}.jsonl").read_text(encoding="utf-8").split(chr(10))
    rows = [l for l in lines if l.strip() and json.loads(l)["id"] in keep]
    assert len(rows) == len(ids)
    (S / f"{ds}_fresh.ids").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (P / f"{ds}_fresh.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(ds, src, a, b, len(ids))
