"""v2 development (dev items 0-299 only; PROTOCOL §17 v2 entry, V1): how often each candidate answer is correct.
Candidates per item: G = v1 final answer (devB2_ragverify), R = standard RAG + reranker answer on the same evidence
(devB_rag_rerank), C = closed-book answer (v1 trace). Prints accuracy of each, agreement rates and oracle bounds.
Usage: .venv/Scripts/python.exe analysis/v2/dev_candidates.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from ragverify import data as D  # noqa: E402
from score import score_example  # noqa: E402


def load(run):
    return {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "results" / run / "outputs.jsonl").open(encoding="utf-8")}


for ds in ["popqa_lt", "triviaqa", "hotpotqa", "pubhealth"]:
    exs = {e["id"]: e for e in D.load(ds, "dev", 300)}
    g, r = load(f"devB2_ragverify_{ds}"), load(f"devB_rag_rerank_{ds}")
    sel = load(f"devB_selfrag_{ds}")
    ok = lambda ex, p: score_example(ex, {"pred": p, "native_abstain": False})["correct"]  # noqa: E731
    n = 0; acc = {"G": 0, "R": 0, "C": 0, "S": 0, "orGR": 0, "orGRC": 0, "agreeGR": 0, "acc_agreeGR": 0, "agreeGC": 0, "acc_agreeGC": 0}
    for i, ex in exs.items():
        if i not in g or i not in r:
            continue
        n += 1
        G, R, C = g[i]["pred"], r[i]["pred"], g[i]["trace"]["cb_ans"]
        cg, cr, cc = ok(ex, G), ok(ex, R), ok(ex, C)
        acc["G"] += cg; acc["R"] += cr; acc["C"] += cc; acc["S"] += ok(ex, sel[i]["pred"]) if i in sel else 0
        acc["orGR"] += max(cg, cr); acc["orGRC"] += max(cg, cr, cc)
        if G.strip().lower() == R.strip().lower():
            acc["agreeGR"] += 1; acc["acc_agreeGR"] += cg
        if G.strip().lower() == C.strip().lower():
            acc["agreeGC"] += 1; acc["acc_agreeGC"] += cg
    print(ds, n, {k: round(v / n, 3) for k, v in acc.items() if not k.startswith("acc_")},
          "acc|agreeGR", round(acc["acc_agreeGR"] / max(1, acc["agreeGR"]), 3), "acc|agreeGC", round(acc["acc_agreeGC"] / max(1, acc["agreeGC"]), 3))
