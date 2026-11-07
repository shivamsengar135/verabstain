"""Build the local Wikipedia index from the BEIR HotpotQA corpus (2017-10 intro paragraphs).

Steps (run in order; each is idempotent):
  prepare : parquet -> passage store (title + text, intros longer than 150 words split into <=100-word chunks)
  bm25    : chunked BM25 (k1=0.9, b=0.4; lowercase, \\w+ tokens, sklearn English stop words, Snowball/Porter stemmer)
  dense   : bge-small-en-v1.5 passage embeddings (fp16, GPU)
  faiss   : IVF-SQ8 inner-product index over the embeddings
Usage: .venv/Scripts/python.exe scripts/build_index.py prepare|bm25|dense|faiss
Outputs in indexes/wiki17/ (see README.txt written there).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "indexes" / "wiki17"
SRC = ROOT / "data" / "raw" / "beir_hotpotqa" / "corpus" / "corpus-00000-of-00001.parquet"
MAX_WORDS, CHUNK_WORDS = 150, 100
K1, B = 0.9, 0.4


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare() -> None:
    import pyarrow.parquet as pq
    OUT.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(SRC)
    blob = (OUT / "passages.bin").open("wb")
    offsets, doc_ids, n_src, n_split, pos = [0], [], 0, 0, 0
    for batch in pf.iter_batches(batch_size=200_000, columns=["_id", "title", "text"]):
        d = batch.to_pydict()
        for did, title, text in zip(d["_id"], d["title"], d["text"]):
            n_src += 1
            text = " ".join((text or "").split())
            if not text:
                continue
            words = text.split(" ")
            chunks = [text] if len(words) <= MAX_WORDS else [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), CHUNK_WORDS)]
            n_split += len(chunks) > 1
            for c in chunks:
                rec = (title.strip() + "\t" + c).encode("utf-8")
                blob.write(rec)
                pos += len(rec)
                offsets.append(pos)
                doc_ids.append(did)
        log(f"prepared {n_src:,} articles -> {len(doc_ids):,} passages")
    blob.close()
    np.save(OUT / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    (OUT / "doc_ids.txt").write_text("\n".join(doc_ids), encoding="utf-8")
    meta = {"source": "BeIR/hotpotqa corpus (HotpotQA 2017-10-01 Wikipedia intro paragraphs)", "articles": n_src,
            "passages": len(doc_ids), "articles_split": n_split, "max_words": MAX_WORDS, "chunk_words": CHUNK_WORDS}
    (OUT / "meta_prepare.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(json.dumps(meta))


class Store:
    def __init__(self, d: Path = OUT):
        self.off = np.load(d / "offsets.npy", mmap_mode="r")
        self.mm = np.memmap(d / "passages.bin", dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        return len(self.off) - 1

    def get(self, i: int) -> tuple[str, str]:
        t, x = bytes(self.mm[self.off[i]:self.off[i + 1]]).decode("utf-8").split("\t", 1)
        return t, x


TOKEN_RE = re.compile(r"\w+")


def make_tokenizer():
    import Stemmer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stem = Stemmer.Stemmer("english")
    stop = frozenset(ENGLISH_STOP_WORDS)

    def tok(text: str) -> list[str]:
        return stem.stemWords([w for w in TOKEN_RE.findall(text.lower()) if w not in stop])
    return tok


def bm25() -> None:
    store, tok = Store(), make_tokenizer()
    n = len(store)
    vocab: dict[str, int] = {}
    parts_t, parts_d, parts_f = [], [], []
    dl = np.zeros(n, dtype=np.int32)
    step = 250_000
    for s in range(0, n, step):
        tids, dids = [], []
        for i in range(s, min(n, s + step)):
            t, x = store.get(i)
            toks = tok(t + " " + x)
            dl[i] = len(toks)
            for w in toks:
                j = vocab.get(w)
                if j is None:
                    j = vocab[w] = len(vocab)
                tids.append(j)
            dids.extend([i] * len(toks))
        key = np.asarray(dids, dtype=np.int64) * (1 << 25) + np.asarray(tids, dtype=np.int64)
        uk, cnt = np.unique(key, return_counts=True)
        parts_d.append((uk >> 25).astype(np.int32))
        parts_t.append((uk & ((1 << 25) - 1)).astype(np.int32))
        parts_f.append(np.minimum(cnt, 65535).astype(np.uint16))
        log(f"bm25 tokenized {min(n, s + step):,}/{n:,} passages, vocab {len(vocab):,}")
    assert len(vocab) < (1 << 25)
    V = len(vocab)
    df = np.zeros(V, dtype=np.int64)
    for t in parts_t:
        df += np.bincount(t, minlength=V)
    nnz = int(df.sum())
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)  # Lucene-style non-negative idf
    avgdl = float(dl.mean())
    indptr = np.zeros(V + 1, dtype=np.int64)
    np.cumsum(df, out=indptr[1:])
    out_d = np.lib.format.open_memmap(OUT / "bm25_docs.npy", mode="w+", dtype=np.int32, shape=(nnz,))
    out_w = np.lib.format.open_memmap(OUT / "bm25_w.npy", mode="w+", dtype=np.float16, shape=(nnz,))
    cursor = indptr[:-1].copy()
    for k in range(len(parts_t)):  # counting sort into term-major postings, doc order preserved
        t, d, f = parts_t[k], parts_d[k], parts_f[k].astype(np.float32)
        w = idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * dl[d] / avgdl))
        o = np.argsort(t, kind="stable")
        t, d, w = t[o], d[o], w[o]
        u, c = np.unique(t, return_counts=True)
        within = np.arange(len(t), dtype=np.int64) - np.repeat(np.cumsum(c) - c, c)
        pos = np.repeat(cursor[u], c) + within
        out_d[pos] = d
        out_w[pos] = w.astype(np.float16)
        cursor[u] += c
        parts_t[k] = parts_d[k] = parts_f[k] = None
    out_d.flush()
    out_w.flush()
    np.save(OUT / "bm25_indptr.npy", indptr)
    (OUT / "bm25_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    meta = {"k1": K1, "b": B, "avgdl": avgdl, "vocab": V, "postings": nnz, "idf": "log(1+(N-df+0.5)/(df+0.5))",
            "tokenizer": "lowercase \\w+, sklearn ENGLISH_STOP_WORDS removed, PyStemmer english", "weights": "float16"}
    (OUT / "meta_bm25.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(json.dumps(meta))


ENCODER = ROOT / "models" / "bge-small-en-v1.5"


def dense() -> None:
    import torch
    from sentence_transformers import SentenceTransformer
    store = Store()
    n = len(store)
    model = SentenceTransformer(str(ENCODER), device="cuda")
    model.half()
    model.max_seq_length = 256
    out = np.lib.format.open_memmap(OUT / "dense_bge_small.npy", mode="r+" if (OUT / "dense_bge_small.npy").exists() else "w+",
                                    dtype=np.float16, shape=(n, 384))
    done_f = OUT / "dense_done.txt"
    start = int(done_f.read_text()) if done_f.exists() else 0
    step = 65_536
    for s in range(start, n, step):
        texts = []
        for i in range(s, min(n, s + step)):
            t, x = store.get(i)
            texts.append(f"{t}. {x}")
        with torch.inference_mode():
            emb = model.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        out[s:s + len(texts)] = emb.astype(np.float16)
        out.flush()
        done_f.write_text(str(s + len(texts)))
        log(f"dense encoded {s + len(texts):,}/{n:,}")
    meta = {"encoder": "BAAI/bge-small-en-v1.5", "passage_format": "{title}. {text}", "max_seq_length": 256,
            "normalized": True, "dtype": "float16", "query_instruction": "Represent this sentence for searching relevant passages: "}
    (OUT / "meta_dense.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def build_faiss(nlist: int = 16384) -> None:
    import faiss
    X = np.load(OUT / "dense_bge_small.npy", mmap_mode="r")
    n, dim = X.shape
    rng = np.random.default_rng(0)
    sample = np.asarray(X[np.sort(rng.choice(n, size=min(n, 400_000), replace=False))], dtype=np.float32)
    quant = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFScalarQuantizer(quant, dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    log("training IVF-SQ8")
    index.train(sample)
    for s in range(0, n, 500_000):
        index.add(np.asarray(X[s:s + 500_000], dtype=np.float32))
        log(f"faiss added {min(n, s + 500_000):,}/{n:,}")
    faiss.write_index(index, str(OUT / "dense_ivf_sq8.faiss"))
    meta = {"index": f"IVF{nlist},SQ8 inner product", "train_sample": int(len(sample)), "seed": 0, "default_nprobe": 128}
    (OUT / "meta_faiss.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    step = sys.argv[1]
    {"prepare": prepare, "bm25": bm25, "dense": dense, "faiss": build_faiss}[step]()
