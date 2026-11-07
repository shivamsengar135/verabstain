"""Score run outputs: per-example correctness and a run summary.

Usage: .venv/Scripts/python.exe scripts/score.py RUN_ID [RUN_ID ...]   (or 'all')
Writes results/<run_id>/scores.jsonl and summary.json. Matching rules:
  * PopQA / TriviaQA: Self-RAG `match` (case-sensitive raw substring; literature/code_refs/selfrag_metrics.py)
    as primary; also normalised match, EM, F1 (SQuAD normalisation)
  * HotpotQA: Self-RAG match primary (sentence-style outputs of Self-RAG would be unfairly penalised by EM); EM, F1 secondary
  * PubHealth: accuracy of the true/false label (Self-RAG match on ['true'] / ['false'])
  * RGB: RGB evalue.py checkanswer (lowercased containment; every answer element must match);
    rejection = native abstention or 'insufficient information' in the output
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragverify import data as D  # noqa: E402
from ragverify.metrics import em, f1, match_norm, match_selfrag  # noqa: E402


def rgb_check(pred: str, gold) -> int:
    p = pred.lower()
    gold = gold if isinstance(gold, list) else [gold]
    labels = []
    for inst in gold:
        if isinstance(inst, list):
            labels.append(int(any(i.lower() in p for i in inst)))
        else:
            labels.append(int(inst.lower() in p))
    return int(0 not in labels)


def score_example(ex: dict, out: dict) -> dict:
    pred = out.get("pred") or ""
    ab = bool(out.get("native_abstain"))
    if ex["dataset"] == "rgb":
        cond = ex["meta"]["condition"]
        rejected = ab or "insufficient information" in pred.lower()
        if cond == "rejection":
            return {"correct": int(rejected), "rejected": int(rejected)}
        gold = ex["answers"]
        if cond == "counterfactual" and len(gold) == 1 and isinstance(gold[0], list):
            # prepare_data.py wrapped en_fact answers in an extra list; RGB's evalue.py passes instance['answer']
            # unchanged, so unwrap it (PROTOCOL §17, 2026-11-06; affects 28 items whose answer is a list of variants)
            gold = gold[0]
        r = {"correct": int((not rejected) and rgb_check(pred, gold)), "rejected": int(rejected)}
        if cond == "counterfactual":
            r["fake_answer"] = int(ex["meta"]["fakeanswer"].lower() in pred.lower())
        return r
    golds = ex["answers"]
    r = {"match": match_selfrag(pred, golds), "match_norm": match_norm(pred, golds), "em": em(pred, golds), "f1": f1(pred, golds)}
    r["correct"] = r["match"]  # primary for every QA dataset (Self-RAG rule); EM/F1 secondary (PROTOCOL §2, decided on dev 11-05)
    return r


def score_run(run_id: str) -> dict:
    d = ROOT / "results" / run_id
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    job = meta["job"]
    exs = {e["id"]: e for e in D.load(job["dataset"], job["split"], job.get("n"))}
    outs = [json.loads(l) for l in (d / "outputs.jsonl").open(encoding="utf-8")]
    rows = []
    for o in outs:
        if o["id"] not in exs:
            continue
        s = score_example(exs[o["id"]], o)
        rows.append({"id": o["id"], **s, "native_abstain": bool(o.get("native_abstain")), "scores": o.get("scores", {}),
                     "cost": o.get("cost", {}), "seconds": o.get("seconds")})
    with (d / "scores.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    summ = {"run_id": run_id, "n": n, "n_expected": len(exs), "method": job["method"], "dataset": job["dataset"], "split": job["split"]}
    for k in ("correct", "match", "match_norm", "em", "f1", "rejected", "fake_answer"):
        vals = [r[k] for r in rows if k in r]
        if vals:
            summ[k] = float(np.mean(vals))
    summ["native_abstain_rate"] = float(np.mean([r["native_abstain"] for r in rows])) if rows else None
    for k in ("llm_calls", "prompt_tokens", "completion_tokens", "llm_seconds", "retrievals", "verifier_pairs"):
        summ[f"mean_{k}"] = float(np.mean([r["cost"].get(k, 0) for r in rows])) if rows else None
    summ["mean_seconds"] = float(np.mean([r["seconds"] for r in rows if r["seconds"] is not None])) if rows else None
    summ["wall_s"] = meta.get("wall_s")
    (d / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    return summ


if __name__ == "__main__":
    ids = sys.argv[1:]
    if ids == ["all"]:
        ids = sorted(p.name for p in (ROOT / "results").iterdir() if (p / "outputs.jsonl").exists())
    for rid in ids:
        s = score_run(rid)
        keys = [k for k in ("correct", "em", "f1", "match", "native_abstain_rate", "mean_llm_calls", "mean_seconds") if s.get(k) is not None]
        print(rid, s["n"], " ".join(f"{k}={s[k]:.3f}" for k in keys))
