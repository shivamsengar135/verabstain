"""Paths, SQLite cache, run statistics and the llama-server client.

Every LLM call goes through LLM.chat / LLM.complete, which (1) looks up a SQLite cache keyed by
sha256(model tag + endpoint + payload), (2) otherwise calls llama-server, and (3) records calls,
tokens and seconds in a thread-safe Stats object that the runner writes to the ledger.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

# Set = LLM requests may be sent. Cleared by the runner's thermal guard; affects timing only, never outputs.
THERMAL_OK = threading.Event()
THERMAL_OK.set()

import requests

ROOT = Path(__file__).resolve().parents[2]
DATA, MODELS, INDEXES = ROOT / "data", ROOT / "models", ROOT / "indexes"
RESULTS, LOGS, CACHE_DIR = ROOT / "results", ROOT / "logs", ROOT / "cache"
LLAMA_DIR = ROOT / "tools" / "llama-b12110"
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("PYTHONUTF8", "1")


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class KVCache:
    """Process/thread-safe key-value store (SQLite WAL, zlib-compressed JSON values)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        con = self._con()
        con.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB, t REAL)")
        con.commit()

    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=120, check_same_thread=False)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            self._local.con = con
        return con

    def get(self, key: str):
        row = self._con().execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return json.loads(zlib.decompress(row[0])) if row else None

    def put(self, key: str, value) -> None:
        blob = zlib.compress(json.dumps(value, ensure_ascii=False).encode("utf-8"), 3)
        con = self._con()
        for i in range(10):
            try:
                con.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?)", (key, blob, time.time()))
                con.commit()
                return
            except sqlite3.OperationalError:
                time.sleep(0.2 * (i + 1))
        raise RuntimeError("cache write failed")


@dataclass
class Stats:
    """Thread-safe counters for one run (LLM calls, tokens, seconds; cache hits counted separately)."""
    llm_calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_seconds: float = 0.0
    retrievals: int = 0
    verifier_pairs: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def as_dict(self) -> dict:
        with self.lock:
            return {k: getattr(self, k) for k in ("llm_calls", "cache_hits", "prompt_tokens", "completion_tokens",
                                                   "llm_seconds", "retrievals", "verifier_pairs")}


class Counter:
    """Per-example cost counter (attached to an example while a method runs)."""

    def __init__(self):
        self.d = {"llm_calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                  "llm_seconds": 0.0, "retrievals": 0, "verifier_pairs": 0}

    def add(self, **kw):
        for k, v in kw.items():
            self.d[k] = self.d.get(k, 0) + v


class LLM:
    """Client for a running llama-server (OpenAI-compatible chat endpoint + native /completion)."""

    def __init__(self, tag: str, base_url: str = "http://127.0.0.1:8091", cache: KVCache | None = None,
                 stats: Stats | None = None, timeout: float = 600):
        self.tag, self.base, self.timeout = tag, base_url.rstrip("/"), timeout
        self.cache = cache or KVCache(CACHE_DIR / "llm.sqlite")
        self.stats = stats or Stats()
        self.session = requests.Session()

    def _post(self, path: str, payload: dict) -> dict:
        THERMAL_OK.wait()  # thermal guard (scripts/run_queue.py) blocks every new LLM request while the laptop cools
        for i in range(8):
            try:
                r = self.session.post(self.base + path, json=payload, timeout=self.timeout)
                if r.status_code == 503:  # loading / busy
                    time.sleep(2 * (i + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(3 * (i + 1))
        raise RuntimeError(f"llama-server unreachable at {self.base}{path}")

    def _cached(self, endpoint: str, payload: dict, counter: Counter | None, fn):
        key = stable_hash({"tag": self.tag, "ep": endpoint, "p": payload})
        hit = self.cache.get(key)
        if hit is not None:
            self.stats.add(cache_hits=1)
            if counter:  # cost of a cached call is still attributed to the example
                counter.add(llm_calls=1, cached_calls=1, prompt_tokens=hit["prompt_tokens"],
                            completion_tokens=hit["completion_tokens"], llm_seconds=hit.get("seconds", 0.0))
            return hit
        t0 = time.perf_counter()
        out = fn()
        out["seconds"] = time.perf_counter() - t0
        self.cache.put(key, out)
        self.stats.add(llm_calls=1, prompt_tokens=out["prompt_tokens"], completion_tokens=out["completion_tokens"],
                       llm_seconds=out["seconds"])
        if counter:
            counter.add(llm_calls=1, prompt_tokens=out["prompt_tokens"], completion_tokens=out["completion_tokens"],
                        llm_seconds=out["seconds"])
        return out

    def chat(self, messages: list[dict], max_tokens: int = 64, temperature: float = 0.0, seed: int = 0,
             top_logprobs: int = 0, stop: list[str] | None = None, counter: Counter | None = None,
             top_p: float = 1.0) -> dict:
        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
                   "top_p": top_p, "stop": stop or [], "logprobs": top_logprobs > 0,
                   "top_logprobs": top_logprobs if top_logprobs > 0 else None, "cache_prompt": True}
        payload = {k: v for k, v in payload.items() if v is not None}

        def fn():
            r = self._post("/v1/chat/completions", payload)
            ch = r["choices"][0]
            toks = []
            for t in ((ch.get("logprobs") or {}).get("content") or []):
                toks.append({"t": t["token"], "lp": t["logprob"],
                             "top": [[x["token"], x["logprob"]] for x in t.get("top_logprobs", [])]})
            u = r.get("usage", {})
            return {"text": ch["message"].get("content") or "", "tokens": toks, "finish": ch.get("finish_reason"),
                    "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0)}
        return self._cached("chat", payload, counter, fn)

    def complete(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0, seed: int = 0,
                 n_probs: int = 0, stop: list[str] | None = None, counter: Counter | None = None) -> dict:
        """Raw-prompt completion via native /completion (used for Self-RAG's own prompt format)."""
        payload = {"prompt": prompt, "n_predict": max_tokens, "temperature": temperature, "seed": seed,
                   "n_probs": n_probs, "stop": stop or [], "cache_prompt": True, "top_k": 0 if temperature == 0 else 40}

        def fn():
            r = self._post("/completion", payload)
            toks = []
            for t in r.get("completion_probabilities") or []:
                top = t.get("top_logprobs") or t.get("probs") or []
                toks.append({"t": t.get("token", ""), "id": t.get("id"), "lp": t.get("logprob"),
                             "top": [[x.get("token"), x.get("logprob", x.get("prob")), x.get("id")] for x in top]})
            return {"text": r.get("content", ""), "tokens": toks, "finish": r.get("stop_type") or r.get("stopped_word"),
                    "prompt_tokens": r.get("tokens_evaluated", 0), "completion_tokens": r.get("tokens_predicted", 0)}
        return self._cached("complete", payload, counter, fn)

    def next_token_logprobs(self, prompt: str, wanted: list[str], counter: Counter | None = None) -> dict:
        """Full-vocabulary log-probabilities of selected next tokens (one-token /completion); caches only `wanted`."""
        payload = {"prompt": prompt, "n_predict": 1, "temperature": 0, "n_probs": 32016, "cache_prompt": True, "wanted": wanted}

        def fn():
            r = self._post("/completion", {k: v for k, v in payload.items() if k != "wanted"})
            top = (r.get("completion_probabilities") or [{}])[0].get("top_logprobs", [])
            d = {x["token"]: x["logprob"] for x in top}
            return {"lp": {w: d.get(w) for w in wanted}, "prompt_tokens": r.get("tokens_evaluated", 0),
                    "completion_tokens": r.get("tokens_predicted", 0), "text": r.get("content", ""), "tokens": []}
        return self._cached("next_token_logprobs", payload, counter, fn)

    def tokenize(self, text: str) -> list[int]:
        return self._post("/tokenize", {"content": text}).get("tokens", [])
