"""Benchmark llama-server on RAG-length prompts at 1/4/8 concurrent requests; check log-probs.

Usage: .venv/Scripts/python.exe scripts/bench_backend.py PROFILE [--ctx 32768 --parallel 8] [--out logs/bench_<profile>.json]
Measures (a) prompt-processing and generation tokens/s reported by the server per request,
(b) aggregate wall-clock throughput, (c) peak VRAM (nvidia-smi), (d) whether chat log-probs and
native /completion n_probs are returned. Unique prompts (random ids) defeat prompt caching.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import llm_server  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORDS = ("river city century museum painter treaty railway election novel bridge island composer battle "
         "university festival harbour dynasty mountain physicist cathedral album senator village league").split()


def rag_prompt(rng: random.Random, n_pass: int = 5, words: int = 120) -> list[dict]:
    passages = []
    for i in range(n_pass):
        body = " ".join(rng.choice(WORDS) for _ in range(words))
        passages.append(f"[{i+1}] Title: Doc {rng.randint(0, 10**9)}\n{body}")
    q = f"Question {rng.randint(0, 10**9)}: which {rng.choice(WORDS)} is described in the passages?"
    return [{"role": "system", "content": "Answer using only the passages. Cite passages like [1]."},
            {"role": "user", "content": "\n\n".join(passages) + "\n\n" + q}]


def vram_used_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return int(out[0]) if out else -1


def one(port: int, msgs: list[dict], max_tokens: int) -> dict:
    t0 = time.perf_counter()
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", timeout=600, json={
        "messages": msgs, "max_tokens": max_tokens, "temperature": 0.0, "logprobs": True, "top_logprobs": 5,
        "cache_prompt": False}).json()
    dt = time.perf_counter() - t0
    tm = r.get("timings", {})
    lp = r["choices"][0].get("logprobs") or {}
    return {"wall": dt, "pp_tps": tm.get("prompt_per_second"), "tg_tps": tm.get("predicted_per_second"),
            "prompt_n": tm.get("prompt_n"), "pred_n": tm.get("predicted_n"), "has_logprobs": bool(lp.get("content"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--parallel", type=int)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out")
    a = ap.parse_args()
    base_vram = vram_used_mb()
    st = llm_server.start(a.profile, ctx=a.ctx, parallel=a.parallel)
    port = st["port"]
    rng = random.Random(0)
    res = {"profile": a.profile, "server": {k: st[k] for k in ("tag", "ctx", "parallel", "load_seconds")},
           "vram_before_mb": base_vram, "vram_loaded_mb": vram_used_mb(), "levels": {}}
    peak = [0]
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], vram_used_mb())
            time.sleep(0.5)
    threading.Thread(target=watch, daemon=True).start()
    one(port, rag_prompt(rng), 8)  # warm-up
    for conc in (1, 4, 8):
        n = conc * 3
        prompts = [rag_prompt(rng) for _ in range(n)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(conc) as ex:
            outs = list(ex.map(lambda m: one(port, m, a.max_tokens), prompts))
        wall = time.perf_counter() - t0
        res["levels"][conc] = {
            "requests": n, "wall_s": round(wall, 2),
            "mean_prompt_tokens": statistics.mean(o["prompt_n"] or 0 for o in outs),
            "mean_gen_tokens": statistics.mean(o["pred_n"] or 0 for o in outs),
            "per_request_pp_tps": round(statistics.median(o["pp_tps"] or 0 for o in outs), 1),
            "per_request_tg_tps": round(statistics.median(o["tg_tps"] or 0 for o in outs), 1),
            "aggregate_gen_tps": round(sum(o["pred_n"] or 0 for o in outs) / wall, 1),
            "aggregate_total_tps": round(sum((o["pred_n"] or 0) + (o["prompt_n"] or 0) for o in outs) / wall, 1),
            "requests_per_s": round(n / wall, 3), "logprobs_ok": all(o["has_logprobs"] for o in outs)}
        print(conc, res["levels"][conc], flush=True)
    # native /completion with n_probs (needed for Self-RAG reflection-token scores)
    r = requests.post(f"http://127.0.0.1:{port}/completion", timeout=120, json={
        "prompt": "The capital of France is", "n_predict": 3, "temperature": 0, "n_probs": 10}).json()
    cp = r.get("completion_probabilities") or []
    res["completion_nprobs_ok"] = bool(cp) and bool(cp[0].get("top_logprobs") or cp[0].get("probs"))
    res["completion_prob_keys"] = sorted(cp[0].keys()) if cp else []
    stop.set()
    res["vram_peak_mb"] = peak[0]
    out = Path(a.out or ROOT / "logs" / f"bench_{a.profile}_c{st['ctx']}_p{st['parallel']}.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("vram_loaded_mb", "vram_peak_mb", "completion_nprobs_ok")}))


if __name__ == "__main__":
    main()
