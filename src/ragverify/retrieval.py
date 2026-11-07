"""Local Wikipedia retrieval: BM25 (memory-mapped postings), dense bge-small + FAISS IVF-SQ8,
reciprocal rank fusion, cross-encoder reranking. Every call is cached in cache/retrieval.sqlite.

Passages are dicts {"id", "title", "text", "src", "score"}; local passages have id "w:<row>".
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np

from .core import CACHE_DIR, INDEXES, MODELS, KVCache, stable_hash

IDX = INDEXES / "wiki17"
QUERY_INSTR = "Represent this sentence for searching relevant passages: "
RRF_K = 60


class Retriever:
    def __init__(self, index_dir: Path = IDX, device: str = "cpu", nprobe: int = 128, load_dense: bool = True):
        self.dir = Path(index_dir)
        self.cache = KVCache(CACHE_DIR / "retrieval.sqlite")
        self.off = np.load(self.dir / "offsets.npy", mmap_mode="r")
        self.mm = np.memmap(self.dir / "passages.bin", dtype=np.uint8, mode="r")
        self.n = len(self.off) - 1
        self._bm25 = None
        self._dense = None
        self._ce = None
        self.device, self.nprobe, self.load_dense = device, nprobe, load_dense
        self.lock = threading.Lock()
        self.version = json.dumps({k: json.loads((self.dir / f"meta_{k}.json").read_text()) for k in
                                   ("prepare", "bm25", "dense", "faiss") if (self.dir / f"meta_{k}.json").exists()},
                                  sort_keys=True)

    # ---------- passages ----------
    def passage(self, row: int) -> dict:
        t, x = bytes(self.mm[self.off[row]:self.off[row + 1]]).decode("utf-8").split("\t", 1)
        return {"id": f"w:{row}", "title": t, "text": x, "src": "local"}

    def get(self, pid: str) -> dict:
        return self.passage(int(pid.split(":", 1)[1]))

    # ---------- BM25 ----------
    def _load_bm25(self):
        if self._bm25 is None:
            import Stemmer
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            vocab = json.loads((self.dir / "bm25_vocab.json").read_text(encoding="utf-8"))
            self._bm25 = {
                "vocab": vocab, "indptr": np.load(self.dir / "bm25_indptr.npy", mmap_mode="r"),
                "docs": np.load(self.dir / "bm25_docs.npy", mmap_mode="r"),
                "w": np.load(self.dir / "bm25_w.npy", mmap_mode="r"),
                "stem": Stemmer.Stemmer("english"), "stop": frozenset(ENGLISH_STOP_WORDS)}
        return self._bm25

    def bm25(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        key = stable_hash({"m": "bm25", "q": query, "k": k, "v": self.version})
        hit = self.cache.get(key)
        if hit is not None:
            return [tuple(x) for x in hit]
        b = self._load_bm25()
        toks = b["stem"].stemWords([w for w in re.findall(r"\w+", query.lower()) if w not in b["stop"]])
        scores: dict | None = None
        acc = np.zeros(self.n, dtype=np.float32)
        touched = []
        for w in toks:  # repeated query terms count repeatedly (as in Lucene's BM25 query)
            j = b["vocab"].get(w)
            if j is None:
                continue
            s, e = int(b["indptr"][j]), int(b["indptr"][j + 1])
            d = np.asarray(b["docs"][s:e])
            acc[d] += np.asarray(b["w"][s:e], dtype=np.float32)
            touched.append(d)
        if not touched:
            res = []
        else:
            cand = np.unique(np.concatenate(touched))
            sc = acc[cand]
            top = np.argsort(-sc, kind="stable")[:k]
            res = [(int(cand[i]), float(sc[i])) for i in top]
        self.cache.put(key, res)
        return res

    # ---------- dense ----------
    def _load_dense(self):
        if self._dense is None:
            import faiss
            from sentence_transformers import SentenceTransformer
            index = faiss.read_index(str(self.dir / "dense_ivf_sq8.faiss"))
            index.nprobe = self.nprobe
            enc = SentenceTransformer(str(MODELS / "bge-small-en-v1.5"), device=self.device)
            if self.device == "cuda":
                enc.half()
            self._dense = {"index": index, "enc": enc}
        return self._dense

    def dense_many(self, queries: list[str], k: int = 50) -> list[list[tuple[int, float]]]:
        keys = [stable_hash({"m": "dense", "q": q, "k": k, "np": self.nprobe, "v": self.version}) for q in queries]
        out = [self.cache.get(kk) for kk in keys]
        todo = [i for i, o in enumerate(out) if o is None]
        if todo:
            with self.lock:
                dn = self._load_dense()
                emb = dn["enc"].encode([QUERY_INSTR + queries[i] for i in todo], batch_size=64,
                                       normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
                D, I = dn["index"].search(emb.astype(np.float32), k)
            for r, i in enumerate(todo):
                res = [(int(p), float(s)) for p, s in zip(I[r], D[r]) if p >= 0]
                self.cache.put(keys[i], res)
                out[i] = res
        return [[tuple(x) for x in o] for o in out]

    def dense(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        return self.dense_many([query], k)[0]

    # ---------- fusion and reranking ----------
    @staticmethod
    def rrf(lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
        sc: dict[str, float] = {}
        for lst in lists:
            for rank, pid in enumerate(lst):
                sc[pid] = sc.get(pid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(sc.items(), key=lambda x: (-x[1], x[0]))

    def hybrid(self, query: str, k_each: int = 50) -> list[tuple[str, float]]:
        bm = [f"w:{p}" for p, _ in self.bm25(query, k_each)]
        dn = [f"w:{p}" for p, _ in self.dense(query, k_each)]
        return self.rrf([bm, dn])

    def _load_ce(self):
        if self._ce is None:
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder(str(MODELS / "msmarco-minilm-l6-v2"), device=self.device, max_length=384)
            if self.device == "cuda":
                self._ce.model.half()
        return self._ce

    def rerank(self, query: str, passages: list[dict]) -> list[float]:
        """Cross-encoder logits for (query, title + text) pairs, cached per pair."""
        keys = [stable_hash({"m": "ce", "q": query, "t": p["title"], "x": p["text"]}) for p in passages]
        out = [self.cache.get(k) for k in keys]
        todo = [i for i, o in enumerate(out) if o is None]
        if todo:
            with self.lock:
                ce = self._load_ce()
                sc = ce.predict([(query, f"{passages[i]['title']}. {passages[i]['text']}") for i in todo],
                                batch_size=32, show_progress_bar=False)
            for j, i in enumerate(todo):
                self.cache.put(keys[i], float(sc[j]))
                out[i] = float(sc[j])
        return out
