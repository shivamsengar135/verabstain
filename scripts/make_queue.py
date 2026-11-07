"""Generate job queues. Usage: .venv/Scripts/python.exe scripts/make_queue.py pilot > jobs/queue_pilot.yaml"""
from __future__ import annotations

import sys

import yaml

DEV = ["popqa_lt", "triviaqa", "hotpotqa", "pubhealth"]


def pilot(n: int = 100) -> list[dict]:
    jobs = []
    llm_only = ["closed_book", "rag_topk", "rag_rerank", "rag_largek", "rag_selfcheck", "chain_of_note", "flare", "cove", "rarr"]
    for m in llm_only:
        for ds in DEV:
            j = {"id": f"pilot_{m}_{ds}", "method": m, "dataset": ds, "split": "dev", "n": n, "server": "qwen7b_q4",
                 "seed": 0, "projected_minutes": 10}
            if m == "closed_book":
                j["needs_retrieval"] = False
            if m == "flare":
                j["config_overrides"] = {"flare_theta": 1.0}
            jobs.append(j)
    for ds in DEV:
        jobs.append({"id": f"pilot_ragverify_{ds}", "method": "ragverify", "dataset": ds, "split": "dev", "n": n,
                     "server": "qwen7b_q4", "seed": 0, "projected_minutes": 15, "verifier": "minicheck-ft5",
                     "config_overrides": {"gate_tau": 1.01, "sup_tau": 1.01}})
    for ds in DEV:
        jobs.append({"id": f"pilot_crag_{ds}", "method": "crag", "dataset": ds, "split": "dev", "n": n, "server": "qwen7b_q4",
                     "seed": 0, "projected_minutes": 10, "config_overrides": {"crag_compute_all": True}})
    for ds in DEV:
        jobs.append({"id": f"pilot_selfrag_{ds}", "method": "selfrag", "dataset": ds, "split": "dev", "n": n,
                     "server": "selfrag7b_q4", "seed": 0, "projected_minutes": 10, "config_overrides": {"selfrag_always_compute": True}})
    return jobs


COST = {"closed_book": 0.05, "rag_topk": 0.34, "rag_rerank": 0.62, "rag_largek": 0.48, "rag_selfcheck": 0.41,
        "chain_of_note": 1.28, "flare": 0.54, "cove": 1.38, "rarr": 3.5, "crag": 1.5, "selfrag": 0.9, "ragverify": 2.0}
SERVER = {"crag": "qwen7b_q4_lean", "ragverify": "qwen7b_q4_lean", "selfrag": "selfrag7b_q4"}
MAIN_DS = [("popqa_lt", 1000), ("triviaqa", 1000), ("hotpotqa", 1000), ("pubhealth", None), ("rgb", None)]
ORDER = ["closed_book", "rag_topk", "rag_rerank", "rag_largek", "rag_selfcheck", "chain_of_note", "flare", "cove", "rarr",
         "crag", "ragverify", "selfrag"]


def job(m, ds, n, seed=0, split="test", prefix="main", backbone=None):
    size = n or {"pubhealth": 987, "rgb": 1300}[ds]  # n given for every non-main dataset
    j = {"id": f"{prefix}_{m}{'' if seed == 0 else f'_s{seed}'}_{ds}", "method": m, "dataset": ds, "split": split, "n": n,
         "server": backbone or SERVER.get(m, "qwen7b_q4"), "seed": seed,
         "projected_minutes": max(3, round(COST[m] * size / 60 * 1.5))}
    if m == "closed_book":
        j["needs_retrieval"] = False
    if m == "ragverify":
        j["verifier"] = "minicheck-ft5"
    if backbone == "llama3b_q4" and m in ("crag", "ragverify"):
        j["server"] = "llama3b_q4"
    return j


def main_queue() -> list[dict]:
    jobs = [job(m, ds, n) for ds, n in MAIN_DS for m in ORDER]
    jobs += [job("rag_selfcheck", ds, n, seed=s) for s in (1, 2) for ds, n in MAIN_DS]
    return jobs


def _rv(name, ds, n, method_args=None, overrides=None, verifier="minicheck-ft5", server="qwen7b_q4_lean", cost=2.0):
    j = {"id": f"abl_ragverify_{name}_{ds}", "method": "ragverify", "dataset": ds, "split": "test", "n": n,
         "server": server, "seed": 0, "projected_minutes": max(3, round(cost * n / 60 * 1.5)), "verifier": verifier}
    if method_args:
        j["method_args"] = method_args
    if overrides:
        j["config_overrides"] = overrides
    return j


def phase4_queue() -> list[dict]:
    """PROTOCOL §10/§14 Phase 4. Specification details fixed before any Phase-4 result exists (PROTOCOL §17, 11-05):
    - ablations: PopQA-LT + HotpotQA, first 500 test items (prefix of the main 1,000-item runs);
      -M1 is the main configuration (gate already off), so the H3 contrast runs '+M1' with gate_tau 0.9;
      rounds=0 is identical to -M5 and is not run twice; -M6 is offline (risk-coverage only).
    - secondary backbone Llama-3.2-3B Q4_K_M: PopQA-LT, HotpotQA, PubHealth x 500; A1 = rag_topk scored offline.
    - quantization check: Llama-3.2-3B Q8_0 vs Q4_K_M, PopQA-LT + HotpotQA first 200, closed-book / RAG top-k /
      RAG-Verify (Q4 numbers are the first 200 items of the secondary-backbone runs)."""
    jobs = []
    for ds in ("popqa_lt", "hotpotqa"):
        n = 500
        jobs += [
            _rv("plusM1g09", ds, n, overrides={"gate_tau": 0.9}),
            _rv("noM2", ds, n, method_args={"m2": False}),
            _rv("noM3", ds, n, method_args={"m3": False}),
            _rv("noM4", ds, n, method_args={"m4": False}),
            _rv("noM5", ds, n, method_args={"m5": False}),
            _rv("selfverify", ds, n, method_args={"self_verify": True}, verifier=None),
            _rv("hhem", ds, n, verifier="hhem-2.1-open"),
            _rv("minicheckdeb", ds, n, verifier="minicheck-deb"),
            _rv("k3", ds, n, overrides={"k": 3}),
            _rv("k10", ds, n, overrides={"k": 10}),
            _rv("rounds1", ds, n, overrides={"rounds": 1}),
        ]
    for j in jobs:
        if j["verifier"] is None:
            del j["verifier"]
    sec = ["closed_book", "rag_topk", "rag_rerank", "flare", "cove", "rag_selfcheck", "crag", "ragverify"]
    for ds in ("popqa_lt", "hotpotqa", "pubhealth"):
        jobs += [job(m, ds, 500, prefix="sec3b", backbone="llama3b_q4") for m in sec]
    for ds in ("popqa_lt", "hotpotqa"):
        for m in ("closed_book", "rag_topk", "ragverify"):
            j = job(m, ds, 200, prefix="quant3bq8", backbone="llama3b_q8")
            j["server"] = "llama3b_q8"
            jobs.append(j)
    return jobs


def distractor_queue() -> list[dict]:
    """PROTOCOL §11 HotpotQA distractor experiment (data: scripts/make_distractor_data.py; §17 entry 11-05 15:10)."""
    jobs = []
    for p in (0, 25, 50, 75):
        for m in ("rag_topk", "rag_rerank", "chain_of_note", "crag", "selfrag", "ragverify"):
            jobs.append(job(m, f"hotpotqa_dist{p:02d}", 300, prefix="rob"))
    return jobs


if __name__ == "__main__":
    which = sys.argv[1]
    print(yaml.safe_dump({"pilot": pilot, "main": main_queue, "phase4": phase4_queue, "distractor": distractor_queue}[which](), sort_keys=False, width=200))
