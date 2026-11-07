"""Offline dev tuning from 'maximal' pilot runs (no extra LLM calls). Dev data only.

Each thresholded method is run once on dev in a maximal mode that records every branch:
  ragverify : gate off (c1 recorded) and sup_tau=1.01 (all revision rounds recorded)
  flare     : theta=1.0 (all re-retrieval rounds recorded with their min token probability)
  crag      : crag_compute_all (answers for all three actions recorded with evaluator scores)
  selfrag   : selfrag_always_compute (no-retrieval and retrieval answers recorded with the ratio)
This script simulates every threshold on a grid, scores it with the dataset's primary rule and picks ONE
global setting per method (mean primary accuracy over the dev datasets; ties -> cheaper setting).
Usage: .venv/Scripts/python.exe analysis/tune_dev.py RUN_ID_PREFIX   (e.g. pilot_)
Writes analysis/dev_tuning.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ragverify import data as D  # noqa: E402
from score import score_example  # noqa: E402


def load_run(rid: str):
    d = ROOT / "results" / rid
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    job = meta["job"]
    exs = {e["id"]: e for e in D.load(job["dataset"], job["split"], job.get("n"))}
    outs = [json.loads(l) for l in (d / "outputs.jsonl").open(encoding="utf-8")]
    return job, exs, outs


def acc(exs, outs, pred_fn) -> float:
    vals = []
    for o in outs:
        p = pred_fn(o)
        vals.append(score_example(exs[o["id"]], {"pred": p, "native_abstain": False})["correct"])
    return float(np.mean(vals)) if vals else float("nan")


RERANK = {}  # id -> rag_rerank prediction (same evidence as RAG-Verify's M2 when rel_tau is off)


def sim_ragverify(o, gate, sup, fb="closed_book"):
    t = o["trace"]
    if t.get("path") == "closed":
        return o["pred"]
    if t["c1"] >= gate and t["cb_ans"].lower() not in ("unknown", ""):
        return t["cb_ans"]
    if t["draft"] == "insufficient evidence" and fb == "rag":  # approximation: no revision rounds on the fallback
        return RERANK.get(o["id"], t["cb_ans"])
    ans = t["cb_ans"] if t["draft"] == "insufficient evidence" else t["draft"]
    s = t.get("support0", 0.0)
    for h in t.get("history", []):
        if s >= sup:
            break
        if h["s"] is not None and h["s"] >= s:
            ans, s = h["ans"], h["s"]
    return ans


def sim_flare(o, theta):
    h = o["trace"]["history"]
    i = 0
    while i < len(h) - 1 and h[i]["minp"] < theta:
        i += 1
    return h[i]["ans"]


def sim_crag(o, up, lo):
    sc = o["trace"]["eval_scores"]
    flags = [2 if s >= up else (1 if s >= lo else 0) for s in sc]
    a = 2 if 2 in flags else (1 if 1 in flags else 0)
    return o["trace"]["all_actions"][str(a)]["pred"]


def sim_selfrag(o, th):
    t = o["trace"]
    return t["pred_ret"] if t["ret_ratio"] > th else t["pred_noret"]


GRIDS = {
    "ragverify": [(g, s, fb) for g in (0.6, 0.7, 0.8, 0.9, 0.95, 1.01) for s in (0.1, 0.3, 0.5, 0.7, 0.9, 1.01)
                  for fb in ("closed_book", "rag")],
    "flare": [(th,) for th in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)],
    "crag": [(u, l) for u in (-1.0, 0.0, 0.5, 1.0, 2.0) for l in (-3.0, -2.0, -1.0, 0.0) if l <= u],
    "selfrag": [(th,) for th in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01)],
}
SIM = {"ragverify": sim_ragverify, "flare": sim_flare, "crag": sim_crag, "selfrag": sim_selfrag}


def main(prefix: str) -> None:
    runs = sorted(p.name for p in (ROOT / "results").iterdir() if p.name.startswith(prefix) and (p / "outputs.jsonl").exists())
    by_method = defaultdict(list)
    for rid in runs:
        job, exs, outs = load_run(rid)
        if job["method"] == "rag_rerank":
            RERANK.update({o["id"]: o["pred"] for o in outs})
        if job["method"] in SIM and job["split"] == "dev":
            by_method[job["method"]].append((job["dataset"], exs, outs))
    res = {}
    for m, items in by_method.items():
        table = []
        for params in GRIDS[m]:
            per = {ds: acc(exs, outs, lambda o: SIM[m](o, *params)) for ds, exs, outs in items}
            table.append({"params": params, "per_dataset": per, "mean": float(np.mean(list(per.values())))})
        best = max(table, key=lambda r: r["mean"])
        res[m] = {"best": best, "grid": table}
        print(m, "best", best["params"], {k: round(v, 3) for k, v in best["per_dataset"].items()}, round(best["mean"], 4))
    (ROOT / "analysis" / "dev_tuning.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
