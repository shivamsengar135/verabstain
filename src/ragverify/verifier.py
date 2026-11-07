"""Claim-level verifiers (RAG-Verify M4 candidates and independent faithfulness evaluators).

Each verifier maps (evidence document, claim) -> support probability in [0, 1].
Inference follows the released code (copies in literature/code_refs/):
  * MiniCheck (Tang et al., 2024): doc + eos + claim; Flan-T5: 'predict: ' prefix, P(label '1') from the
    first decoder step over token ids [3, 209]; RoBERTa/DeBERTa: 2-way classifier, softmax[:, 1].
    Docs are chunked by sentences (FT5: 500 words; encoders: 400 tokens); score = max over chunks.
  * AlignScore-large (Zha et al., 2023) 'nli_sp': premise chunks of ~350 words, claim split into
    sentences, tri-label head softmax[:, 0] (aligned), max over chunks then mean over sentences.
  * HHEM-2.1-Open (Vectara): its own predict() on (premise, hypothesis).
  * DeBERTa-v3-large NLI (MoritzLaurer): P(entailment).
Deviation: sentence splitting uses a regex splitter instead of NLTK punkt (documented in PROTOCOL.md).
Scores are cached in cache/verifier.sqlite.
"""
from __future__ import annotations

import re
import threading

import numpy as np
import torch

from .core import CACHE_DIR, MODELS, KVCache, stable_hash

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def sent_split(text: str) -> list[str]:
    out = []
    for block in text.split("\n"):
        block = block.strip()
        if block:
            out.extend(s.strip() for s in _SENT.split(block) if s.strip())
    return out or [""]


class Verifier:
    name = "base"
    family = "base"

    def __init__(self, device: str = "cuda", batch_size: int = 16):
        self.device, self.bs = device, batch_size
        self.cache = KVCache(CACHE_DIR / "verifier.sqlite")
        self.lock = threading.Lock()
        self.model = None

    def _score_uncached(self, docs: list[str], claims: list[str]) -> list[float]:
        raise NotImplementedError

    def score(self, docs: list[str], claims: list[str]) -> list[float]:
        keys = [stable_hash({"v": self.name, "d": d, "c": c}) for d, c in zip(docs, claims)]
        out = [self.cache.get(k) for k in keys]
        todo = [i for i, o in enumerate(out) if o is None]
        if todo:
            with self.lock:
                res = self._score_uncached([docs[i] for i in todo], [claims[i] for i in todo])
            for i, r in zip(todo, res):
                self.cache.put(keys[i], float(r))
                out[i] = float(r)
        return out

    def unload(self) -> None:
        self.model = None
        torch.cuda.empty_cache()


class MiniCheck(Verifier):
    family = {"ft5": "t5", "deb": "deberta", "rbt": "roberta"}

    def __init__(self, variant: str = "ft5", **kw):
        super().__init__(**kw)
        self.variant = variant
        self.name = f"minicheck-{variant}"
        self.family = MiniCheck.family[variant]
        self.path = MODELS / {"ft5": "minicheck-ft5-l", "deb": "minicheck-deb-l", "rbt": "minicheck-rbt-l"}[variant]
        self.max_len = {"ft5": 2048, "deb": 2048, "rbt": 512}[variant]
        self.chunk = 500 if variant == "ft5" else 400

    def _load(self):
        if self.model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(str(self.path))
            if self.variant == "ft5":
                m = AutoModelForSeq2SeqLM.from_pretrained(str(self.path), dtype=torch.bfloat16)
            else:
                m = AutoModelForSequenceClassification.from_pretrained(str(self.path), num_labels=2, dtype=torch.float16)
            self.model = m.to(self.device).eval()

    def _chunks(self, doc: str) -> list[str]:
        sents, chunks, cur, cnt = sent_split(doc), [], [], 0
        for s in sents:
            n = len(s.split()) if self.variant == "ft5" else len(self.tok(s, add_special_tokens=False)["input_ids"])
            if cur and cnt + n > self.chunk:
                chunks.append(" ".join(cur))
                cur, cnt = [s], n
            else:
                cur.append(s)
                cnt += n
        if cur:
            chunks.append(" ".join(cur))
        return [c for c in chunks if c.strip()] or [doc]

    @torch.inference_mode()
    def _score_uncached(self, docs, claims):
        self._load()
        pairs, owner = [], []
        for i, (d, c) in enumerate(zip(docs, claims)):
            for ch in self._chunks(d):
                pairs.append(ch + self.tok.eos_token + c)
                owner.append(i)
        probs = []
        for s in range(0, len(pairs), self.bs):
            batch = pairs[s:s + self.bs]
            if self.variant == "ft5":
                enc = self.tok(["predict: " + t for t in batch], max_length=self.max_len, truncation=True,
                               padding=True, return_tensors="pt").to(self.device)
                dec = torch.zeros((enc["input_ids"].size(0), 1), dtype=torch.long, device=self.device)
                logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                    decoder_input_ids=dec).logits[:, 0, :]
                p = torch.softmax(logits[:, torch.tensor([3, 209], device=self.device)].float(), dim=-1)[:, 1]
            else:
                enc = self.tok(batch, max_length=self.max_len, truncation=True, padding=True, return_tensors="pt").to(self.device)
                p = torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, 1]
            probs.extend(p.cpu().tolist())
        out = np.zeros(len(docs))
        for i, p in zip(owner, probs):
            out[i] = max(out[i], p)
        return out.tolist()


class AlignScoreLarge(Verifier):
    name, family = "alignscore-large", "roberta"

    def _load(self):
        if self.model is None:
            from transformers import AutoConfig, AutoTokenizer, RobertaModel
            tokdir = MODELS / "roberta-large-tok"
            self.tok = AutoTokenizer.from_pretrained(str(tokdir))
            base = RobertaModel(AutoConfig.from_pretrained(str(tokdir)))
            ck = torch.load(str(MODELS / "alignscore" / "AlignScore-large.ckpt"), map_location="cpu", weights_only=False)
            sd = ck["state_dict"]
            base.load_state_dict({k[len("base_model."):]: v for k, v in sd.items() if k.startswith("base_model.")}, strict=False)
            tri = torch.nn.Linear(base.config.hidden_size, 3)
            tri.load_state_dict({k[len("tri_layer."):]: v for k, v in sd.items() if k.startswith("tri_layer.")})
            self.model = base.half().to(self.device).eval()
            self.tri = tri.half().to(self.device).eval()
            del ck, sd

    @torch.inference_mode()
    def _score_uncached(self, docs, claims):
        self._load()
        out = []
        for d, c in zip(docs, claims):
            psents = sent_split(d)
            n_chunk = len(d.strip().split()) // 350 + 1
            n_chunk = max(len(psents) // n_chunk, 1)
            pchunks = [" ".join(psents[i:i + n_chunk]) for i in range(0, len(psents), n_chunk)]
            hsents = sent_split(c)
            prem = [p for p in pchunks for _ in hsents]
            hyp = [h for _ in pchunks for h in hsents]
            scores = []
            for s in range(0, len(prem), self.bs):
                enc = self.tok(prem[s:s + self.bs], hyp[s:s + self.bs], truncation="only_first", max_length=512,
                               padding="max_length", return_tensors="pt").to(self.device)
                pooled = self.model(**enc).pooler_output
                scores.extend(torch.softmax(self.tri(pooled).float(), dim=-1)[:, 0].cpu().tolist())
            m = np.asarray(scores).reshape(len(pchunks), len(hsents))
            out.append(float(m.max(axis=0).mean()))
        return out


class HHEM(Verifier):
    name, family = "hhem-2.1-open", "t5"

    def _load(self):
        if self.model is None:
            from transformers import AutoModelForSequenceClassification
            m = AutoModelForSequenceClassification.from_pretrained(str(MODELS / "hhem-2.1-open"), trust_remote_code=True)
            self.model = m.to(self.device).eval()

    @torch.inference_mode()
    def _score_uncached(self, docs, claims):
        self._load()
        out = []
        for s in range(0, len(docs), self.bs):
            out.extend(self.model.predict(list(zip(docs[s:s + self.bs], claims[s:s + self.bs]))).float().cpu().tolist())
        return out


class DebertaNLI(Verifier):
    name, family = "deberta-v3-l-nli", "deberta"

    def _load(self):
        if self.model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            p = str(MODELS / "deberta-v3-l-nli")
            self.tok = AutoTokenizer.from_pretrained(p)
            self.model = AutoModelForSequenceClassification.from_pretrained(p, dtype=torch.float16).to(self.device).eval()
            self.ent = {v.lower(): k for k, v in self.model.config.id2label.items()}["entailment"]

    @torch.inference_mode()
    def _score_uncached(self, docs, claims):
        self._load()
        out = []
        for s in range(0, len(docs), self.bs):
            enc = self.tok(docs[s:s + self.bs], claims[s:s + self.bs], truncation="only_first", max_length=512,
                           padding=True, return_tensors="pt").to(self.device)
            out.extend(torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, self.ent].cpu().tolist())
        return out


def make(name: str, **kw) -> Verifier:
    if name.startswith("minicheck-"):
        return MiniCheck(name.split("-", 1)[1], **kw)
    return {"alignscore-large": AlignScoreLarge, "hhem-2.1-open": HHEM, "deberta-v3-l-nli": DebertaNLI}[name](**kw)
