"""Load processed examples in the fixed evaluation order (data/splits/<name>_<split>.ids)."""
from __future__ import annotations

import json

from .core import DATA


def load(name: str, split: str, n: int | None = None, offset: int = 0) -> list[dict]:
    path = DATA / "processed" / f"{name}_{split}.jsonl"
    order = [l for l in (DATA / "splits" / f"{name}_{split}.ids").read_text(encoding="utf-8").splitlines() if l.strip()]
    want = order[offset: offset + n] if n else order[offset:]
    keep = set(want)
    byid = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["id"] in keep:
                byid[e["id"]] = e
    return [byid[i] for i in want]
