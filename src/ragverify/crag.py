"""CRAG (Yan et al., 2024) re-implemented on our backbone and local index.

Follows the released code (literature/code_refs/crag_*.py):
  * retrieval evaluator: released T5-large T5ForSequenceClassification(num_labels=1), input
    'question [SEP] passage', max_length 512, score = logit
  * per-document flag: score >= upper -> 2 (correct), >= lower -> 1 (ambiguous), else 0; action =
    Correct if any 2, else Ambiguous if any 1, else Incorrect
  * internal knowledge (Correct/Ambiguous): decompose passages into strips ('excerption' mode), score
    each strip with the evaluator, keep the top 6, join with '; '
  * external knowledge (Incorrect/Ambiguous): keyword query from the backbone with CRAG's few-shot
    prompt; Google search replaced by our local index (same search() as every other method);
    strips from the returned passages, top 5 by evaluator score
  * Ambiguous: 'Knowledge1: <internal> [sep] Knowledge2: <external>' (the evident intent of
    combined_knowledge_preparation.py, whose released loop mis-uses enumerate)
Thresholds are tuned on dev with the same budget as other methods (see PROTOCOL.md).
"""
from __future__ import annotations

import threading

import torch

from . import prompts as P
from .core import CACHE_DIR, MODELS, KVCache, stable_hash
from .methods import clean_answer, gm_prob, is_unknown


def extract_strips(psg: str, mode: str = "excerption") -> list[str]:
    """Copy of CRAG extract_strips_from_psg (excerption / selection)."""
    if mode == "selection":
        return [psg]
    num_concatenate_strips = 3
    question_strips = psg.split("?")
    origin_strips = []
    for qs in question_strips:
        origin_strips += qs.split(". ")
    strips = []
    for s in origin_strips:
        if s in strips:
            continue
        if not strips:
            strips.append(s)
        elif len(s.split()) > 5:
            strips.append(s)
        else:
            strips[-1] += s
    final, buf = [], []
    for strip in strips:
        buf.append(strip)
        if len(buf) == num_concatenate_strips:
            final.append(" ".join(buf))
            buf = []
    if buf:
        final.append(" ".join(buf))
    return final


class CragEvaluator:
    def __init__(self, device: str = "cuda", batch_size: int = 4):
        self.device, self.bs = device, batch_size
        self.cache = KVCache(CACHE_DIR / "crag_eval.sqlite")
        self.lock = threading.Lock()
        self.model = None

    def _load(self):
        if self.model is None:
            from transformers import T5ForSequenceClassification, T5Tokenizer
            path = MODELS / "crag-evaluator"
            sub = [p for p in path.rglob("config.json")]
            mdir = sub[0].parent if sub else path
            self.tok = T5Tokenizer.from_pretrained(str(mdir))
            self.model = T5ForSequenceClassification.from_pretrained(str(mdir), num_labels=1, dtype=torch.bfloat16).to(self.device).eval()  # bf16 to fit beside the LLM (8 GB)

    def score(self, query: str, texts: list[str]) -> list[float]:
        keys = [stable_hash({"crag": query, "t": t}) for t in texts]
        out = [self.cache.get(k) for k in keys]
        todo = [i for i, o in enumerate(out) if o is None]
        if todo:
            with self.lock, torch.inference_mode():
                self._load()
                for s in range(0, len(todo), self.bs):
                    idx = todo[s:s + self.bs]
                    enc = self.tok([query + " [SEP] " + texts[i] for i in idx], return_tensors="pt", padding="max_length",
                                   truncation=True, max_length=512).to(self.device)
                    logits = self.model(enc["input_ids"], attention_mask=enc["attention_mask"]).logits.view(-1).float().cpu().tolist()
                    for i, v in zip(idx, logits):
                        self.cache.put(keys[i], v)
                        out[i] = v
        return out

    def select(self, query: str, strips: list[str], top_n: int) -> str:
        """Copy of CRAG select_relevants: strips with < 4 words get -1; top_n by score joined with '; '."""
        long = [s for s in strips if len(s.split()) >= 4]
        sc = dict(zip(long, self.score(query, long))) if long else {}
        ranked = sorted(((sc.get(s, -1.0), s) for s in strips), key=lambda x: x[0], reverse=True)
        return "; ".join(s for _, s in ranked[:top_n]).replace("\n", " ")


class CRAG:
    name = "crag"

    def __init__(self, ctx, **kw):
        self.ctx = ctx

    def run(self, ex, cnt):
        c, ev = self.ctx.cfg, self.ctx.crag_eval
        docs = self.ctx.static_list(ex, c["k"], cnt)
        q = ex["question"]
        scores = ev.score(q, [p["text"] for p in docs])
        flags = [2 if s >= c["crag_upper"] else (1 if s >= c["crag_lower"] else 0) for s in scores]
        action = 2 if 2 in flags else (1 if 1 in flags else 0)
        tr = {"eval_scores": scores, "action": ["incorrect", "ambiguous", "correct"][action], "pids": [p["id"] for p in docs]}
        internal = external = ""
        all_mode = c.get("crag_compute_all", False)  # dev 'maximal' mode: every action computed for offline threshold tuning
        if action >= 1 or all_mode:
            strips = [s for p in docs for s in extract_strips(p["text"], "excerption")]
            internal = ev.select(q, strips, 6)
        if action <= 1 or all_mode:
            kw = self.ctx.llm.chat(P.crag_keywords(ex), max_tokens=c["max_query_tokens"], counter=cnt)["text"].strip().split("\n")[0]
            tr["keywords"] = kw
            web = self.ctx.search(ex, kw or q, c["k"], cnt)
            strips = [s for p in web for s in extract_strips(p["text"], "excerption")]
            external = ev.select(q, strips, 5)
            tr["web_pids"] = [p["id"] for p in web]
        know = {2: internal, 0: external, 1: f"Knowledge1: {internal} [sep] Knowledge2: {external}"}
        outs = {}
        for a in ([0, 1, 2] if all_mode else [action]):
            r = self.ctx.llm.chat(P.crag_generate(ex, know[a]), max_tokens=c["max_answer_tokens"], top_logprobs=5, counter=cnt)
            outs[a] = (clean_answer(r["text"], ex["task"]), gm_prob(r["tokens"]))
        if all_mode:
            tr["all_actions"] = {str(a): {"pred": p, "tokenprob": s} for a, (p, s) in outs.items()}
        pred, tp = outs[action]
        return {"pred": pred, "native_abstain": is_unknown(pred), "scores": {"tokenprob": tp,
                "crag_max_eval": max(scores) if scores else -99.0}, "trace": tr}
