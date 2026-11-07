"""Baselines and RAG-Verify. Every method maps one example -> one output record:
  pred         final answer text used for accuracy with abstention off
  native_abstain  the method itself declined ("unknown" / "insufficient evidence")
  scores       confidence scores used for risk-coverage analysis (higher = more confident)
  trace        intermediate artefacts (passage ids, drafts, verifier scores, queries) for error analysis
Costs are accumulated in the per-example Counter by the LLM / retrieval / verifier wrappers.
"""
from __future__ import annotations

import math
import re

import numpy as np

from . import prompts as P
from .core import Counter
from .metrics import match_norm, normalize_answer

# ---------------------------------------------------------------- helpers

def clean_answer(text: str, task: str) -> str:
    t = (text or "").strip()
    m = re.findall(r"answer\s*:\s*(.+)", t, flags=re.I)
    if m:
        t = m[-1]
    t = t.strip().split("\n")[0].strip().strip('"').strip("*").strip()
    t = re.sub(r"[\s.]+$", "", t)
    if task == "claim":
        mm = re.search(r"\b(true|false)\b", t.lower()) or re.search(r"\b(true|false)\b", (text or "").lower())
        return mm.group(1) if mm else "unknown"
    return t


def gm_prob(tokens: list[dict], start: int = 0, end: int | None = None) -> float:
    lps = [t["lp"] for t in tokens[start:end] if t.get("lp") is not None]
    return float(math.exp(sum(lps) / len(lps))) if lps else 0.0


def answer_span_conf(tokens: list[dict]) -> float:
    """Geometric-mean probability of the tokens on the 'Answer:' line (whole output if no marker)."""
    text, start, end = "", None, None
    for i, t in enumerate(tokens):
        text += t["t"]
        if start is None and re.search(r"answer\s*:\s*$", text, re.I):
            start = i + 1
        elif start is not None and "\n" in t["t"]:
            end = i
            break
    return gm_prob(tokens, start or 0, end)


def yes_prob(tokens: list[dict]) -> float:
    if not tokens:
        return 0.0
    py = pn = 0.0
    for tok, lp in tokens[0]["top"]:
        w = tok.strip().lower()
        if w in ("yes", "true"):
            py += math.exp(lp)
        elif w in ("no", "false"):
            pn += math.exp(lp)
    return py / (py + pn) if py + pn > 0 else 0.5


def parse_grounded(text: str, task: str) -> tuple[str, str]:
    ans = clean_answer(text, task)
    m = re.search(r"(statement|evidence)\s*:\s*(.+)", text or "", re.I)
    stmt = re.sub(r"\[\d+\]", "", m.group(2)).strip() if m else ""
    if P.INSUFFICIENT in (text or "").lower():
        ans = P.INSUFFICIENT
    return ans, stmt


def template_statement(ex: dict, ans: str) -> str:
    return f"The answer to the question \"{ex['question']}\" is {ans}."


def evidence_doc(passages: list[dict]) -> str:
    return "\n".join(f"{p['title']}: {p['text']}" for p in passages)


def is_unknown(ans: str) -> bool:
    a = normalize_answer(ans)
    return a in ("unknown", "insufficient evidence", "i dont know", "") or "insufficient evidence" in a


class Ctx:
    """Shared resources for a run: LLM client, retriever, verifier, global config."""

    def __init__(self, llm, retr, cfg: dict, verifier=None, crag_eval=None, seed: int = 0):
        self.llm, self.retr, self.cfg, self.verifier, self.crag_eval, self.seed = llm, retr, cfg, verifier, crag_eval, seed

    # static first-stage list: released pre-retrieved passages if available, else local dense
    def static_list(self, ex: dict, k: int, cnt: Counter) -> list[dict]:
        cnt.add(retrievals=1)
        if ex.get("ctxs"):
            return [dict(p, id=p.get("id") or f"r:{ex['id']}:{i}", src="released") for i, p in enumerate(ex["ctxs"])][:k]
        how = self.cfg.get("static_retriever", "dense")
        if how == "bm25":
            hits = [pid for pid, _ in self.retr.bm25(ex["question"], max(k, 50))]
        elif how == "hybrid":
            hits = [int(pid.split(":")[1]) for pid, _ in self.retr.hybrid(ex["question"], 50)]
        else:
            hits = [pid for pid, _ in self.retr.dense(ex["question"], max(k, 50))]
        return [self.retr.passage(pid) for pid in hits[:k]]

    # candidate pool for reranking (M2 and the '+reranker' baseline): RRF(static, BM25, dense) -> CE rerank
    def pool(self, ex: dict, cnt: Counter) -> list[dict]:
        cfg = self.cfg
        n = cfg["pool_each"]
        static = self.static_list(ex, n, cnt)
        byid = {p["id"]: p for p in static}
        lists = [[p["id"] for p in static]]
        if not ex.get("corpus"):
            bm = [f"w:{pid}" for pid, _ in self.retr.bm25(ex["question"], n)]
            dn = [f"w:{pid}" for pid, _ in self.retr.dense(ex["question"], max(n, 50))[:n]]
            lists += [bm, dn]
        fused = self.retr.rrf(lists)[: cfg["pool_size"]]
        cands = [byid.get(pid) or self.retr.get(pid) for pid, _ in fused]
        ce = self.retr.rerank(ex["question"], cands)
        for p, s in zip(cands, ce):
            p["ce"] = s
        return sorted(cands, key=lambda p: -p["ce"])

    # follow-up queries issued by dynamic methods (FLARE, CRAG, CoVe, RARR, RAG-Verify M5): same function for all
    def search(self, ex: dict, query: str, k: int, cnt: Counter, exclude: set | None = None) -> list[dict]:
        cnt.add(retrievals=1)
        exclude = exclude or set()
        if ex.get("corpus"):  # benchmark-provided passage pool (e.g. RGB): search within it
            cands = [dict(p, id=p.get("id") or f"c:{ex['id']}:{i}") for i, p in enumerate(ex["corpus"])]
        else:
            fused = self.retr.hybrid(query, self.cfg["pool_each"])[: self.cfg["pool_size"]]
            cands = [self.retr.get(pid) for pid, _ in fused]
        cands = [p for p in cands if p["id"] not in exclude]
        if not cands:
            return []
        ce = self.retr.rerank(query, cands)
        for p, s in zip(cands, ce):
            p["ce"] = s
        return sorted(cands, key=lambda p: -p["ce"])[:k]

    def verify(self, doc: str, claim: str, cnt: Counter) -> float:
        cnt.add(verifier_pairs=1)
        return self.verifier.score([doc], [claim])[0]


# ---------------------------------------------------------------- baselines

class ClosedBook:
    name = "closed_book"

    def __init__(self, ctx: Ctx, **kw):
        self.ctx = ctx

    def run(self, ex, cnt):
        r = self.ctx.llm.chat(P.closed_book(ex), max_tokens=self.ctx.cfg["max_answer_tokens"], top_logprobs=5, counter=cnt)
        pred = clean_answer(r["text"], ex["task"])
        return {"pred": pred, "native_abstain": is_unknown(pred), "scores": {"tokenprob": gm_prob(r["tokens"])},
                "trace": {"raw": r["text"]}}


class StandardRAG:
    """Top-k RAG. variant: 'topk' (static list), 'rerank' (same pool + reranker as RAG-Verify), 'largek'."""

    def __init__(self, ctx: Ctx, variant: str = "topk", **kw):
        self.ctx, self.variant = ctx, variant
        self.name = {"topk": "rag_topk", "rerank": "rag_rerank", "largek": "rag_largek"}[variant]

    def passages(self, ex, cnt):
        c = self.ctx.cfg
        if self.variant == "topk":
            return self.ctx.static_list(ex, c["k"], cnt)
        if self.variant == "largek":
            return self.ctx.static_list(ex, c["k_large"], cnt)
        return self.ctx.pool(ex, cnt)[: c["k"]]

    def run(self, ex, cnt):
        ps = self.passages(ex, cnt)
        r = self.ctx.llm.chat(P.standard_rag(ex, ps), max_tokens=self.ctx.cfg["max_answer_tokens"], top_logprobs=5, counter=cnt)
        pred = clean_answer(r["text"], ex["task"])
        return {"pred": pred, "native_abstain": is_unknown(pred), "scores": {"tokenprob": gm_prob(r["tokens"])},
                "trace": {"pids": [p["id"] for p in ps], "raw": r["text"]}}


class SelfCheckRAG(StandardRAG):
    """Abstention baseline: standard top-k RAG answer + SelfCheckGPT-style sampling consistency
    (fraction of S sampled answers that agree with the greedy answer, normalised containment)."""
    name = "rag_selfcheck"

    def __init__(self, ctx, **kw):
        super().__init__(ctx, "topk")
        self.name = "rag_selfcheck"

    def run(self, ex, cnt):
        out = super().run(ex, cnt)
        ps = self.passages(ex, Counter())
        greedy = out["pred"]
        agree = []
        for j in range(self.ctx.cfg["selfcheck_samples"]):
            r = self.ctx.llm.chat(P.standard_rag(ex, ps), max_tokens=self.ctx.cfg["max_answer_tokens"], temperature=1.0,
                                  seed=1000 * self.ctx.seed + j, counter=cnt)
            s = clean_answer(r["text"], ex["task"])
            agree.append(int(match_norm(s, [greedy]) or match_norm(greedy, [s])) if greedy and s else 0)
        out["scores"]["selfcheck"] = float(np.mean(agree))
        out["trace"]["agree"] = agree
        return out


class ChainOfNote:
    name = "chain_of_note"

    def __init__(self, ctx, **kw):
        self.ctx = ctx

    def run(self, ex, cnt):
        ps = self.ctx.static_list(ex, self.ctx.cfg["k"], cnt)
        r = self.ctx.llm.chat(P.chain_of_note(ex, ps), max_tokens=self.ctx.cfg["max_reasoning_tokens"], top_logprobs=5, counter=cnt)
        pred = clean_answer(r["text"], ex["task"])
        return {"pred": pred, "native_abstain": is_unknown(pred), "scores": {"tokenprob": answer_span_conf(r["tokens"])},
                "trace": {"pids": [p["id"] for p in ps], "raw": r["text"][-600:]}}


class FLARE:
    """Short-form adaptation of FLARE-direct: retrieve with the input, generate the answer with token
    probabilities; if any token has probability < theta, query = question + answer with low-probability
    tokens masked, retrieve again and regenerate (at most `flare_rounds` times)."""
    name = "flare"

    def __init__(self, ctx, **kw):
        self.ctx = ctx

    def run(self, ex, cnt):
        c = self.ctx.cfg
        ps = self.ctx.static_list(ex, c["k"], cnt)
        seen = {p["id"] for p in ps}
        r = self.ctx.llm.chat(P.standard_rag(ex, ps), max_tokens=c["max_answer_tokens"], top_logprobs=5, counter=cnt)
        trace, rounds, hist = {"queries": []}, 0, []
        while True:
            probs = [math.exp(t["lp"]) for t in r["tokens"] if t.get("lp") is not None]
            minp = min(probs) if probs else 1.0
            hist.append({"ans": clean_answer(r["text"], ex["task"]), "minp": minp, "tokenprob": gm_prob(r["tokens"]),
                         "calls": cnt.d["llm_calls"], "ptok": cnt.d["prompt_tokens"], "ctok": cnt.d["completion_tokens"]})
            if rounds >= c["flare_rounds"] or minp >= c["flare_theta"]:
                break
            kept = "".join(t["t"] for t in r["tokens"] if t.get("lp") is not None and math.exp(t["lp"]) >= c["flare_beta"])
            q = f"{ex['question']} {kept}".strip()
            trace["queries"].append(q)
            new = self.ctx.search(ex, q, c["k"], cnt, exclude=set())
            ps = new
            seen |= {p["id"] for p in new}
            r = self.ctx.llm.chat(P.standard_rag(ex, ps), max_tokens=c["max_answer_tokens"], top_logprobs=5, counter=cnt)
            rounds += 1
        pred = clean_answer(r["text"], ex["task"])
        trace.update({"rounds": rounds, "pids": [p["id"] for p in ps], "raw": r["text"], "history": hist})
        return {"pred": pred, "native_abstain": is_unknown(pred), "scores": {"tokenprob": gm_prob(r["tokens"])}, "trace": trace}


class CoVe:
    """Chain-of-Verification, factored: baseline (= top-k RAG answer), plan up to 3 verification
    questions, answer each independently with its own retrieval, final revised answer."""
    name = "cove"

    def __init__(self, ctx, **kw):
        self.ctx = ctx

    def run(self, ex, cnt):
        c = self.ctx.cfg
        base = StandardRAG(self.ctx, "topk").run(ex, cnt)["pred"]
        plan = self.ctx.llm.chat(P.cove_plan(ex, base), max_tokens=c["max_plan_tokens"], counter=cnt)["text"]
        qs = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", l).strip() for l in plan.split("\n")]
        qs = [q for q in qs if q.endswith("?")][:3]
        qa = []
        for q in qs:
            ps = self.ctx.search(ex, q, c["k_sub"], cnt)
            a = self.ctx.llm.chat(P.cove_execute(q, ps), max_tokens=c["max_plan_tokens"], counter=cnt)["text"].strip()
            qa.append((q, a))
        r = self.ctx.llm.chat(P.cove_final(ex, base, qa), max_tokens=c["max_answer_tokens"], top_logprobs=5, counter=cnt)
        pred = clean_answer(r["text"], ex["task"])
        return {"pred": pred, "native_abstain": is_unknown(pred), "scores": {"tokenprob": gm_prob(r["tokens"])},
                "trace": {"baseline": base, "qa": qa}}


# ---------------------------------------------------------------- RAG-Verify

class RAGVerify:
    """M1 gate -> M2 hybrid+rerank+filter -> M3 grounded draft -> M4 external claim verification ->
    M5 targeted re-retrieval and minimal revision (<= rounds) ; M6 abstention is applied offline on
    the recorded support score (risk control needs a calibration split, see analysis/)."""

    def __init__(self, ctx, m1=True, m2=True, m3=True, m4=True, m5=True, self_verify=False, name="ragverify", **kw):
        self.ctx, self.m1, self.m2, self.m3, self.m4, self.m5, self.sv = ctx, m1, m2, m3, m4, m5, self_verify
        self.name = name

    def support(self, ex, ans, stmt, ev, cnt):
        doc = evidence_doc(ev)
        if ex["task"] == "claim":
            if self.sv:
                r = self.ctx.llm.chat(P.self_verify(ex["question"], ev), max_tokens=2, top_logprobs=10, counter=cnt)
                s_claim = yes_prob(r["tokens"])
            else:
                s_claim = self.ctx.verify(doc, ex["question"], cnt)
            return (s_claim if ans == "true" else 1.0 - s_claim), s_claim
        claim = template_statement(ex, ans) if self.ctx.cfg.get("rv_claim", "model") == "template" else (stmt or template_statement(ex, ans))
        if self.sv:
            r = self.ctx.llm.chat(P.self_verify(claim, ev), max_tokens=2, top_logprobs=10, counter=cnt)
            return yes_prob(r["tokens"]), None
        return self.ctx.verify(doc, claim, cnt), None

    def run(self, ex, cnt):
        c, llm = self.ctx.cfg, self.ctx.llm
        tr = {}
        # M1: closed-book draft + confidence (identical payload to the closed-book baseline)
        cb = llm.chat(P.closed_book(ex), max_tokens=c["max_answer_tokens"], top_logprobs=5, counter=cnt)
        cb_ans, c1 = clean_answer(cb["text"], ex["task"]), gm_prob(cb["tokens"])
        tr["cb_ans"], tr["c1"] = cb_ans, c1
        if self.m1 and c1 >= c["gate_tau"] and not is_unknown(cb_ans):
            return {"pred": cb_ans, "native_abstain": False, "scores": {"c1": c1, "path": "closed"},
                    "trace": dict(tr, path="closed")}
        # M2: evidence selection
        if self.m2:
            pool = self.ctx.pool(ex, cnt)
            ev = [p for p in pool[: c["k"]] if p["ce"] >= c["rel_tau"]]
        else:
            ev = self.ctx.static_list(ex, c["k"], cnt)
        tr["pids"] = [p["id"] for p in ev]
        # M3: grounded draft (or the standard RAG prompt when M3 is ablated)
        if ev:
            msgs = P.grounded_draft(ex, ev) if self.m3 else P.standard_rag(ex, ev)
            r = llm.chat(msgs, max_tokens=c["max_draft_tokens"], top_logprobs=5, counter=cnt)
            ans, stmt = parse_grounded(r["text"], ex["task"]) if self.m3 else (clean_answer(r["text"], ex["task"]), "")
            tr["draft_raw"] = r["text"]
            conf_draft = answer_span_conf(r["tokens"])
        else:
            ans, stmt, conf_draft = P.INSUFFICIENT, "", 0.0
        insufficient = ans == P.INSUFFICIENT
        tr["draft"], tr["stmt"] = ans, stmt
        if not self.m4:  # no verification -> no revision; confidence = draft token probability
            pred = cb_ans if insufficient else ans
            return {"pred": pred, "native_abstain": insufficient, "scores": {"c1": c1, "tokenprob": conf_draft, "path": "rag"},
                    "trace": tr}
        if insufficient:
            # fallback for accuracy-with-abstention-off (dev-chosen): closed-book draft, or the forced standard-RAG
            # answer on the same evidence (identical call to the '+reranker' baseline, so it is cache-shared)
            if c.get("rv_fallback", "closed_book") == "rag" and ev:
                rf = llm.chat(P.standard_rag(ex, ev), max_tokens=c["max_answer_tokens"], top_logprobs=5, counter=cnt)
                ans_eval, stmt = clean_answer(rf["text"], ex["task"]), ""
                tr["fallback_rag"] = ans_eval
                conf_draft = gm_prob(rf["tokens"])
            else:
                ans_eval, stmt = cb_ans, ""
        else:
            ans_eval = ans
        if insufficient and c.get("rv_fallback", "closed_book") != "rag":
            conf_draft = c1
        conf_final = conf_draft
        s, s_claim = self.support(ex, ans_eval, stmt, ev, cnt) if ev else (0.0, None)
        tr["support0"] = s
        tr["cost_base"] = dict(cnt.d)
        # M5: targeted re-retrieval + minimal revision for unsupported answers
        rounds, seen, history = 0, {p["id"] for p in ev}, []
        while self.m5 and s < c["sup_tau"] and rounds < c["rounds"]:
            target = ex["question"] if ex["task"] == "claim" else (stmt or template_statement(ex, ans_eval))
            q = llm.chat(P.focused_query(ex, target), max_tokens=c["max_query_tokens"], counter=cnt)["text"].strip().split("\n")[0]
            q = q.strip('"') or ex["question"]
            new = self.ctx.search(ex, q, c["k_sub"], cnt, exclude=seen)
            seen |= {p["id"] for p in new}
            ev2 = (ev + new)[: c["max_evidence"]]
            r = llm.chat(P.revise(ex, ev2, ans_eval), max_tokens=c["max_draft_tokens"], top_logprobs=5, counter=cnt)
            a2, st2 = parse_grounded(r["text"], ex["task"])
            rounds += 1
            if a2 == P.INSUFFICIENT:
                history.append({"q": q, "ans": a2, "s": None, "cost": dict(cnt.d)})
                ev = ev2
                continue
            s2, _ = self.support(ex, a2, st2, ev2, cnt)
            history.append({"q": q, "ans": a2, "s": s2, "cost": dict(cnt.d)})
            if s2 >= s:  # keep the revision only if it is better supported
                ans_eval, stmt, s, ev, insufficient = a2, st2, s2, ev2, False
                conf_final = answer_span_conf(r["tokens"])
        tr.update({"rounds": rounds, "history": history, "final_pids": [p["id"] for p in ev], "s_claim": s_claim})
        return {"pred": ans_eval, "native_abstain": insufficient, "scores": {"c1": c1, "support": s, "path": "rag",
                "tokenprob": conf_final, "tokenprob_draft": conf_draft},
                "trace": tr}


REGISTRY = {
    "closed_book": lambda ctx, **kw: ClosedBook(ctx, **kw),
    "rag_topk": lambda ctx, **kw: StandardRAG(ctx, "topk", **kw),
    "rag_rerank": lambda ctx, **kw: StandardRAG(ctx, "rerank", **kw),
    "rag_largek": lambda ctx, **kw: StandardRAG(ctx, "largek", **kw),
    "rag_selfcheck": lambda ctx, **kw: SelfCheckRAG(ctx, **kw),
    "chain_of_note": lambda ctx, **kw: ChainOfNote(ctx, **kw),
    "flare": lambda ctx, **kw: FLARE(ctx, **kw),
    "cove": lambda ctx, **kw: CoVe(ctx, **kw),
    "ragverify": lambda ctx, **kw: RAGVerify(ctx, **kw),
}
