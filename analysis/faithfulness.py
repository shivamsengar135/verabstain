"""Faithfulness (PROTOCOL §8): unsupported-answer rate = fraction of answered QA items whose answer statement is not
supported (score < 0.5) by the evidence the method used. Two evaluators from families different from M4
(MiniCheck-Flan-T5): AlignScore-large (RoBERTa) and DeBERTa-v3-large NLI.

Answer statement: methods.template_statement(question, prediction), the same template RAG-Verify's M4 uses.
Answered: items without native abstention (abstention off). QA test sets: PopQA-LT, TriviaQA, HotpotQA, RGB.
Evidence used, per method (from each run's trace; text of 'w:' ids from the wiki index, other ids from the item's own
released passages / RGB corpus):
  B2/B3/B4/B6/B7 trace.pids (passages in the final answer prompt; FLARE: last round's passages)
  B5 CRAG: the passages its knowledge strips came from (correct: internal pids; incorrect: web_pids; ambiguous: both)
  B8 Self-RAG: trace.best_pid when it retrieved (otherwise no evidence -> counted as unsupported)
  B9 CoVe: B2's passages for the same item (its baseline answer is B2's); B10 RARR: B2's passages + passages of its
  revision steps; RAG-Verify: trace.final_pids (initial + focused retrieval).
B1 (closed-book) and A1/A2 (B2's answers) are not evaluated separately (no evidence / identical answers to B2).
Thermal safety: before each batch the script waits while the GPU is >= 80 C or the CPU thermal zone >= 78 C (resume at
68 C / 65 C), the same limits as the runner's guard. Run it only when no runner is active.
Usage: .venv/Scripts/python.exe analysis/faithfulness.py -> results/analysis/faithfulness.json, paper/generated/table_faithfulness.tex
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from ragverify import data as D  # noqa: E402
from ragverify import verifier as V  # noqa: E402
from ragverify.methods import evidence_doc, template_statement  # noqa: E402
from main_stats import DS_NAME  # noqa: E402
from method_name import METHOD_NAME  # noqa: E402

DATASETS = ["popqa_lt", "triviaqa", "hotpotqa", "rgb"]
METHODS = [(METHOD_NAME, "ragverify"), ("B2 RAG top-k", "rag_topk"), ("B3 RAG + reranker", "rag_rerank"),
           ("B4 RAG large-k", "rag_largek"), ("B5 CRAG", "crag"), ("B6 Chain-of-Note", "chain_of_note"), ("B7 FLARE", "flare"),
           ("B8 Self-RAG", "selfrag"), ("B9 CoVe", "cove"), ("B10 RARR", "rarr")]
EVALUATORS = ["alignscore-large", "deberta-v3-l-nli"]
BATCH = 64
off = np.load(ROOT / "indexes" / "wiki17" / "offsets.npy", mmap_mode="r")
mm = np.memmap(ROOT / "indexes" / "wiki17" / "passages.bin", dtype=np.uint8, mode="r")


def temps() -> tuple[float, float]:
    try:
        g = float(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20).stdout.strip())
    except Exception:  # noqa: BLE001
        g = 999.0
    try:
        cmd = ("((Get-Counter '\\Thermal Zone Information(*)\\Temperature').CounterSamples | Where-Object "
               "{ $_.InstanceName -like '*tsz0*' } | Select-Object -First 1).CookedValue")
        z = float(subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                                 timeout=30).stdout.strip()) - 273.15
    except Exception:  # noqa: BLE001
        z = 999.0
    return g, z


def wait_cool() -> None:
    g, z = temps()
    if g < 80 and z < 78:
        return
    print(f"thermal pause gpu={g:.0f} cpu_zone={z:.1f}", flush=True)
    while not (g <= 68 and z <= 65):
        time.sleep(10)
        g, z = temps()


def wiki(pid: str) -> dict:
    row = int(pid.split(":", 1)[1])
    t, x = bytes(mm[off[row]:off[row + 1]]).decode("utf-8").split("\t", 1)
    return {"title": t, "text": x}


def outputs(m: str, ds: str) -> dict:
    return {o["id"]: o for o in map(json.loads, (ROOT / "results" / f"main_{m}_{ds}" / "outputs.jsonl").open(encoding="utf-8"))}


def evidence_pids(m: str, o: dict, b2: dict) -> list[str]:
    t = o["trace"]
    if m == "ragverify":
        return t.get("final_pids") or t.get("pids", [])
    if m == "crag":
        return {"correct": t["pids"], "incorrect": t.get("web_pids", []), "ambiguous": t["pids"] + t.get("web_pids", [])}[t["action"]]
    if m == "selfrag":
        return [t["best_pid"]] if t.get("do_retrieve") and t.get("best_pid") else []
    if m == "cove":
        return b2[o["id"]]["trace"]["pids"]
    if m == "rarr":
        return b2[o["id"]]["trace"]["pids"] + [s["pid"] for s in t.get("steps", []) if s.get("pid")]
    return t.get("pids", [])


def build(ds: str) -> dict[str, list[tuple[str, str, str]]]:
    exs = {e["id"]: e for e in D.load(ds, "test", 1000 if ds != "rgb" else None)}
    b2 = outputs("rag_topk", ds)
    items = {}
    for _, m in METHODS:
        rows = []
        for i, o in outputs(m, ds).items():
            if o.get("native_abstain") or i not in exs:
                continue
            ex = exs[i]
            local = {c["id"]: c for c in ex.get("ctxs", []) + ex.get("corpus", [])}
            ps = []
            for pid in dict.fromkeys(evidence_pids(m, o, b2)):
                p = local.get(pid) or (wiki(pid) if pid.startswith("w:") else None)
                if p:
                    ps.append(p)
            rows.append((i, evidence_doc(ps) if ps else "", template_statement(ex, o.get("pred") or "")))
        items[m] = rows
    return items


def main() -> None:
    out_path = ROOT / "results" / "analysis" / "faithfulness.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    work = {ds: build(ds) for ds in DATASETS}
    for ev in EVALUATORS:
        if all(res.get(ev, {}).get(ds, {}).get(m) for ds in DATASETS for _, m in METHODS):
            continue
        model = V.make(ev, device="cuda", batch_size=16)
        for ds in DATASETS:
            for _, m in METHODS:
                if res.get(ev, {}).get(ds, {}).get(m):
                    continue
                rows = work[ds][m]
                scores = []
                for s in range(0, len(rows), BATCH):
                    wait_cool()
                    chunk = rows[s:s + BATCH]
                    with_doc = [(d, c) for _, d, c in chunk if d]
                    sc = iter(model.score([d for d, _ in with_doc], [c for _, c in with_doc]) if with_doc else [])
                    scores += [next(sc) if d else 0.0 for _, d, _ in chunk]
                sc = np.array(scores)
                res.setdefault(ev, {}).setdefault(ds, {})[m] = {
                    "n_answered": len(rows), "unsupported_rate": float((sc < 0.5).mean()) if len(sc) else None,
                    "mean_score": float(sc.mean()) if len(sc) else None, "n_no_evidence": sum(1 for r in rows if not r[1])}
                out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
                print(ev, ds, m, res[ev][ds][m], flush=True)
        model.unload()
    table(res)


def table(res: dict) -> None:
    L = ["% generated by analysis/faithfulness.py: unsupported-answer rate among answered items (score < 0.5); "
         "AlignScore-large / DeBERTa-v3-large NLI",
         "\\begin{tabular}{@{}l" + "c" * len(DATASETS) + "@{}}", "\\toprule Method & " + " & ".join(DS_NAME[d] for d in DATASETS) + " \\\\ \\midrule"]
    for name, m in METHODS:
        cells = []
        for d in DATASETS:
            a, b = res.get(EVALUATORS[0], {}).get(d, {}).get(m), res.get(EVALUATORS[1], {}).get(d, {}).get(m)
            cells.append(f"{a['unsupported_rate']:.3f} / {b['unsupported_rate']:.3f}" if a and b else "--")
        L.append(f"{name} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper" / "generated" / "table_faithfulness.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
