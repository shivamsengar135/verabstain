"""Answer recall@k of BM25, dense and hybrid first-stage retrieval on the dev sets (dev only; no LLM).

Used once to choose `static_retriever` (PROTOCOL §3). Recall@k = fraction of questions whose gold answer
string (normalised containment) appears in at least one of the top-k passages. PubHealth claims have no
answer strings and are skipped. Usage: .venv/Scripts/python.exe analysis/retrieval_recall_dev.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragverify import data as D  # noqa: E402
from ragverify.metrics import normalize_answer  # noqa: E402
from ragverify.retrieval import Retriever  # noqa: E402

retr = Retriever(device="cuda")
res = {}
for ds in ["popqa_lt", "triviaqa", "hotpotqa"]:
    exs = D.load(ds, "dev")
    qs = [e["question"] for e in exs]
    dense = retr.dense_many(qs, 50)
    out = {"bm25": [], "dense": [], "hybrid": []}
    for e, dn in zip(exs, dense):
        golds = [normalize_answer(a) for a in e["answers"] if normalize_answer(a)]
        bm = [p for p, _ in retr.bm25(e["question"], 50)]
        hy = [int(p.split(":")[1]) for p, _ in retr.rrf([[f"w:{p}" for p in bm[:20]], [f"w:{p}" for p, _ in dn[:20]]])]
        for name, lst in (("bm25", bm), ("dense", [p for p, _ in dn]), ("hybrid", hy)):
            txt = [normalize_answer(retr.passage(p)["title"] + " " + retr.passage(p)["text"]) for p in lst[:5]]
            out[name].append(int(any(g in t for g in golds for t in txt)))
    res[ds] = {k: sum(v) / len(v) for k, v in out.items()}
    print(ds, {k: round(v, 3) for k, v in res[ds].items()}, flush=True)
mean = {k: sum(res[d][k] for d in res) / len(res) for k in ("bm25", "dense", "hybrid")}
res["mean"] = mean
print("mean", {k: round(v, 3) for k, v in mean.items()})
(ROOT / "analysis" / "retrieval_recall_dev.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
