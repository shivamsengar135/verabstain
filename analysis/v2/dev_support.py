"""v2 development (dev items 0-299, QA datasets): MiniCheck-FT5 support of each candidate answer on the M2 evidence.
Candidates: D = v1 draft (or its standard-RAG fallback when the draft says 'insufficient'), R = standard RAG + reranker
answer (devB_rag_rerank, same evidence), C = closed-book answer (v1 trace). Claim = v1's template statement.
Output: results/analysis/v2_dev_support.json. GPU (verifier only), with the thermal wait of analysis/faithfulness.py.
Usage: .venv/Scripts/python.exe analysis/v2/dev_support.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import faithfulness as F  # noqa: E402
from ragverify import verifier as V  # noqa: E402
from ragverify.methods import evidence_doc, template_statement  # noqa: E402
from ragverify import data as D  # noqa: E402

OUT = ROOT / "results" / "analysis" / "v2_dev_support.json"


def load(run):
    return {o["id"]: o for o in map(json.loads, (ROOT / "results" / run / "outputs.jsonl").open(encoding="utf-8"))}


def main():
    model = V.make("minicheck-ft5", device="cuda", batch_size=16)
    res = {}
    for ds in ["popqa_lt", "triviaqa", "hotpotqa"]:
        exs = {e["id"]: e for e in D.load(ds, "dev", 300)}
        g, r = load(f"devB2_ragverify_{ds}"), load(f"devB_rag_rerank_{ds}")
        items, docs, claims = [], [], []
        for i, ex in exs.items():
            t = g[i]["trace"]
            doc = evidence_doc([F.wiki(p) for p in t["pids"]])
            d = t.get("fallback_rag") if t["draft"].startswith("insufficient") or t["draft"] == "" else t["draft"]
            d = d if d is not None else g[i]["pred"]
            cand = {"D": d, "R": r[i]["pred"], "C": t["cb_ans"]}
            items.append((i, cand, t))
            for k in "DRC":
                docs.append(doc); claims.append(template_statement(ex, cand[k]))
        sc = []
        for s in range(0, len(docs), 64):
            F.wait_cool()
            sc += model.score(docs[s:s + 64], claims[s:s + 64])
        out = {}
        for n, (i, cand, t) in enumerate(items):
            out[i] = {"cand": cand, "sup": dict(zip("DRC", map(float, sc[3 * n:3 * n + 3]))), "c1": t["c1"],
                      "insufficient": t["draft"] != cand["D"], "tp_r": r[i]["scores"].get("tokenprob")}
        res[ds] = out
        print(ds, len(out), flush=True)
    OUT.write_text(json.dumps(res), encoding="utf-8")
    model.unload()


if __name__ == "__main__":
    main()
