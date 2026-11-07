"""Self-RAG 7B (Asai et al., 2024) with its released checkpoint (own backbone: Llama-2-7B fine-tune),
served as a Q4_K_M GGUF through llama-server's native /completion endpoint.

Follows retrieval_lm/run_short_form.py (call_model_rerank_w_scores_batch), copied in literature/code_refs:
  prompt "### Instruction:\n{instruction}\n\n### Response:\n"; per passage append
  "[Retrieval]<paragraph>{title}\n{text}</paragraph>"; relevance from the first generated token,
  groundedness / utility from the first position where such a token is generated;
  final = exp(seqscore) + w_rel*rel + w_sup*grd + w_use*ut (w = 1, 1, 0.5); best path wins.
Adaptations (recorded in PROTOCOL.md): top-20 token probabilities for relevance/groundedness/utility
instead of 5,000 (tokens outside the top-20 get probability 0, as missing tokens get logprob -100 in the
original); the retrieval decision uses the full vocabulary as in the original;
adaptive-retrieval rule selectable on dev between the paper's normalised probability ('prob') and
the released code's ratio of log-probabilities ('logprob_ratio').
"""
from __future__ import annotations

import math

from .methods import clean_answer, is_unknown

CONTROL = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
           "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]",
           "[Utility:4]", "[Utility:5]"]
REL = ["[Irrelevant]", "[Relevant]"]
GRD = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
UT = [f"[Utility:{i}]" for i in range(1, 6)]
FEVER_INST = "Is the following statement correct or not? Say true if it's correct; otherwise say false."


def postprocess(ans: str) -> str:
    for t in CONTROL:
        ans = ans.replace(t, "")
    return ans.replace("</s>", "").replace("\n", "").replace("<|endoftext|>", "")


def top_probs(tok: dict) -> dict[str, float]:
    out = {}
    for t, lp, *_ in tok["top"]:
        if t is None or lp is None:
            continue
        out[t] = math.exp(lp) if lp <= 0 else lp  # llama-server returns logprob (<=0) or prob depending on mode
    return out


class SelfRAG:
    name = "selfrag"

    def __init__(self, ctx, **kw):
        self.ctx = ctx

    def prompt(self, ex):
        inst = f"{FEVER_INST}\n\n## Input:\n\n{ex['question']}" if ex["task"] == "claim" else ex["question"]
        return f"### Instruction:\n{inst}\n\n### Response:\n"

    def run(self, ex, cnt):
        c, llm = self.ctx.cfg, self.ctx.llm
        mx = c["selfrag_max_tokens_claim"] if ex["task"] == "claim" else c["selfrag_max_tokens_qa"]
        prompt = self.prompt(ex)
        r0 = llm.complete(prompt, max_tokens=mx, n_probs=c["selfrag_nprobs"], counter=cnt)
        lp = llm.next_token_logprobs(prompt, ["[Retrieval]", "[No Retrieval]"], counter=cnt)["lp"]  # full vocabulary, as original
        p_ret = math.exp(lp["[Retrieval]"]) if lp.get("[Retrieval]") is not None else 0.0
        p_no = math.exp(lp["[No Retrieval]"]) if lp.get("[No Retrieval]") is not None else 0.0
        if c["selfrag_ret_rule"] == "prob":
            ratio = p_ret / (p_ret + p_no) if p_ret + p_no > 0 else 0.0
        else:  # released code: ratio of log-probabilities (missing token -> -100)
            lr = math.log(p_ret) if p_ret > 0 else -100.0
            ln = math.log(p_no) if p_no > 0 else -100.0
            ratio = lr / (lr + ln)
        do_ret = ratio > c["selfrag_threshold"]
        pred0 = clean_answer(postprocess(r0["text"]), ex["task"])
        tr = {"ret_ratio": ratio, "do_retrieve": do_ret, "pred_noret": pred0, "p_ret": p_ret, "p_noret": p_no}
        if not do_ret and not c.get("selfrag_always_compute", False):
            return {"pred": pred0, "native_abstain": is_unknown(pred0), "scores": {"seqprob": self.seqprob(r0)}, "trace": tr}
        docs = self.ctx.static_list(ex, c["k"], cnt)
        best = None
        for p in docs:
            r = llm.complete(prompt + f"[Retrieval]<paragraph>{p['title']}\n{p['text']}</paragraph>", max_tokens=mx,
                             n_probs=c["selfrag_nprobs"], counter=cnt)
            toks = r["tokens"]
            rel_p = top_probs(toks[0]) if toks else {}
            rel = rel_p.get("[Relevant]", 0.0) / (sum(rel_p.get(t, 0.0) for t in REL) or 1.0)
            grd = ut = 0.0
            for i, t in enumerate(toks):
                if t["t"] in GRD:
                    d = top_probs(t)
                    s = sum(d.get(x, 0.0) for x in GRD)
                    if s > 0:  # tokens outside the top-n count as probability ~0 (original: logprob -100)
                        grd = d.get("[Fully supported]", 0.0) / s + 0.5 * d.get("[Partially supported]", 0.0) / s
                    break
            for i, t in enumerate(toks):
                if t["t"] in UT:
                    d = top_probs(t)
                    s = sum(d.get(x, 0.0) for x in UT)
                    if s > 0:
                        ut = sum(w * d.get(x, 0.0) / s for w, x in zip([-1, -0.5, 0, 0.5, 1], UT))
                    break
            final = self.seqprob(r) + c["selfrag_w_rel"] * rel + c["selfrag_w_sup"] * grd + c["selfrag_w_use"] * ut
            if best is None or final > best[0]:
                best = (final, r["text"], p["id"], rel, grd, ut)
        pred_ret = clean_answer(postprocess(best[1]), ex["task"])
        tr.update({"best_pid": best[2], "rel": best[3], "grd": best[4], "ut": best[5], "pids": [p["id"] for p in docs],
                   "pred_ret": pred_ret, "final_ret": best[0], "seqprob_noret": self.seqprob(r0)})
        pred = pred_ret if do_ret else pred0
        return {"pred": pred, "native_abstain": is_unknown(pred),
                "scores": {"selfrag_final": best[0] if do_ret else self.seqprob(r0)}, "trace": tr}

    @staticmethod
    def seqprob(r: dict) -> float:
        lps = [t["lp"] for t in r["tokens"] if t.get("lp") is not None]
        return math.exp(sum(lps) / max(len(lps), 1)) if lps else 0.0
