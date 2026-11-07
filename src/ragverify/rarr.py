"""RARR (Gao et al., 2023) re-implemented on our backbone and local index.

Prompts are the released few-shot completion prompts (literature/code_refs/rarr_rarr_prompts.py), read with
ast (not executed). Pipeline as in run_editor_sequential.py: contextual question generation (3 sampled
rounds, temperature 0.7, union of questions) -> one evidence passage per question (the released code keeps
the cross-encoder-best passage of the search results; we use the shared search(), which also reranks with
an MS MARCO MiniLM cross-encoder) -> agreement gate (temperature 0) -> editor when the gate opens.
Adaptations: the edited text is the RAG draft turned into a statement ("The answer to <q> is <a>."), the
question is the context; questions capped at 3 (first-seen order) to bound cost; a final shared extraction
call maps the revised statement to a short answer (same call as other sentence-output methods).
"""
from __future__ import annotations

import ast

from . import prompts as P
from .core import ROOT
from .methods import StandardRAG, clean_answer, gm_prob, is_unknown, template_statement


def _load_prompts() -> dict:
    tree = ast.parse((ROOT / "literature" / "code_refs" / "rarr_rarr_prompts.py").read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        v = node.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            out[node.targets[0].id] = v.value
        elif (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "strip" and not v.args
              and isinstance(v.func.value, ast.Constant) and isinstance(v.func.value.value, str)):  # """...""".strip()
            out[node.targets[0].id] = v.func.value.value.strip()
    return out


PROMPTS = _load_prompts()


class RARR:
    name = "rarr"

    def __init__(self, ctx, **kw):
        self.ctx = ctx

    def run(self, ex, cnt):
        c, llm = self.ctx.cfg, self.ctx.llm
        draft = StandardRAG(self.ctx, "topk").run(ex, cnt)["pred"]
        claim = template_statement(ex, draft) if ex["task"] == "qa" else f"The statement \"{ex['question']}\" is {draft}."
        context = ex["question"]
        qs: list[str] = []
        for j in range(c["rarr_qgen_rounds"]):
            r = llm.complete(PROMPTS["CONTEXTUAL_QGEN_PROMPT"].format(context=context, claim=claim).strip(), max_tokens=256,
                             temperature=c.get("rarr_qgen_temperature", 0.7), seed=1000 * self.ctx.seed + j, counter=cnt)
            for line in r["text"].split("\n"):
                if "I googled:" in line:
                    q = line.split("I googled:")[1].strip()
                    if q and q not in qs:
                        qs.append(q)
        qs = qs[: c["rarr_max_questions"]]
        steps = []
        for q in qs:
            ev = self.ctx.search(ex, q, 1, cnt)
            if not ev:
                continue
            evid = ev[0]["text"]
            g = llm.complete(PROMPTS["CONTEXTUAL_AGREEMENT_GATE_PROMPT"].format(context=context, claim=claim, query=q, evidence=evid).strip(),
                             max_tokens=256, stop=["\n\n"], counter=cnt)["text"].strip().split("\n")
            is_open = len(g) >= 2 and "disagrees" in g[1]
            step = {"q": q, "pid": ev[0]["id"], "gate_open": is_open}
            if is_open:
                e = llm.complete(PROMPTS["CONTEXTUAL_EDITOR_PROMPT"].format(context=context, claim=claim, query=q, evidence=evid).strip(),
                                 max_tokens=256, stop=["\n\n"], counter=cnt)["text"].strip().split("\n")
                if len(e) >= 2:
                    edited = e[1].split("My fix:")[-1].strip()
                    if edited:
                        claim = edited
                step["claim"] = claim
            steps.append(step)
        r = llm.chat(P.extract_short(ex, claim), max_tokens=c["max_answer_tokens"], top_logprobs=5, counter=cnt)
        pred = clean_answer(r["text"], ex["task"])
        return {"pred": pred, "native_abstain": is_unknown(pred), "scores": {"tokenprob": gm_prob(r["tokens"])},
                "trace": {"draft": draft, "questions": qs, "steps": steps, "final_claim": claim}}
