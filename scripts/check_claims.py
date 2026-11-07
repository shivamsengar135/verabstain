"""Rule 7 full-text check: for each statement the manuscript makes about a cited paper, search that paper's full text
(literature/pdfs/<arxiv id>.txt, from scripts/fetch_cited_fulltext.py) for the supporting passage and print it.
The verdict for each claim is recorded by hand in literature/claims_check.csv after reading the printed passages.
Usage: .venv/Scripts/python.exe scripts/check_claims.py [key ...]
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# key -> (claim as stated in the paper, regexes to locate support in the full text)
CLAIMS: dict[str, tuple[str, list[str]]] = {
    "ji2023halsurvey": ("survey of hallucination: generated content that is nonsensical or unfaithful", [r"hallucinat\w* .{0,80}(unfaithful|nonsensical)"]),
    "huang2025halsurvey": ("survey of hallucination in LLMs", [r"survey", r"hallucination"]),
    "lewis2020rag": ("RAG conditions generation on retrieved passages", [r"retriev\w+ .{0,60}(passages|documents)", r"non-parametric memory"]),
    "gao2023ragsurvey": ("survey of RAG", [r"survey"]),
    "mallen2023popqa": ("retrieval helps most for less popular / long-tail entities", [r"less popular", r"long[- ]tail", r"popular(ity)? .{0,60}retriev"]),
    "chen2024rgb": ("noisy/irrelevant passages hurt; negative rejection; counterfactual robustness", [r"noise robustness", r"negative rejection", r"counterfactual robustness"]),
    "liu2024lost": ("models fail to use relevant information in long contexts (middle position)", [r"middle of (the )?(input|long) context", r"performance .{0,40}degrades"]),
    "yoran2024retrobust": ("irrelevant context can hurt RAG", [r"irrelevant (context|evidence)", r"(hurt|harm|degrade)"]),
    "jiang2023flare": ("FLARE retrieves when the upcoming sentence contains low-probability tokens", [r"low[- ](probability|confidence) tokens?", r"upcoming sentence"]),
    "su2024dragin": ("DRAGIN: dynamic retrieval triggered by the model's real-time information needs", [r"real-time information needs", r"when to retrieve"]),
    "wang2023skr": ("SKR: decide retrieval by self-knowledge", [r"self-knowledge", r"when to retrieve|adaptively call"]),
    "yao2024seakr": ("SeaKR: self-aware uncertainty from internal states triggers retrieval", [r"self-aware", r"internal states"]),
    "jeong2024adaptiverag": ("Adaptive-RAG: query-complexity classifier selects strategy", [r"classifier", r"complexity"]),
    "moskvoretskii2025adaptive": ("compares uncertainty estimation methods for adaptive retrieval", [r"uncertainty estimation", r"adaptive retrieval"]),
    "targ2025": ("TARG: training-free gate from signals of a draft generated without context", [r"training-free", r"(no-context|without (retrieved )?context|draft)", r"margin|entropy"]),
    "asai2024selfrag": ("Self-RAG: reflection tokens to decide retrieval and judge relevance/support; released eval files", [r"reflection tokens", r"(IsRel|ISREL|IsSup|ISSUP|Retrieve)"]),
    "yan2024crag": ("CRAG: lightweight evaluator; correct/incorrect/ambiguous -> keep/discard (web search)/both", [r"lightweight retrieval evaluator", r"\{?Correct, Incorrect, Ambiguous\}?|Ambiguous", r"web search"]),
    "yu2024chainofnote": ("Chain-of-Note writes reading notes for each retrieved document before answering", [r"reading notes", r"sequential reading notes"]),
    "gao2023rarr": ("RARR researches and revises an output", [r"research and revis", r"post-edit|revis\w+ the output"]),
    "zhao2023verifyedit": ("Verify-and-Edit edits reasoning chains with retrieved external knowledge", [r"reasoning chains?", r"external knowledge"]),
    "kang2024ever": ("EVER: real-time verification and rectification during generation", [r"real-time", r"rectif"]),
    "dhuliawala2024cove": ("CoVe: plan verification questions, answer them (factored variant)", [r"verification questions", r"factored"]),
    "gou2024critic": ("CRITIC: tool-interactive critiquing", [r"tool-interactive", r"critiqu"]),
    "huang2024cannotselfcorrect": ("LLMs struggle to self-correct reasoning without external feedback", [r"without external feedback", r"intrinsic self-correction"]),
    "min2023factscore": ("FActScore breaks text into atomic facts", [r"atomic facts?"]),
    "wei2024safe": ("SAFE splits responses into individual facts and checks them with search", [r"individual facts", r"Google Search|search queries"]),
    "zha2023alignscore": ("AlignScore: document-grounded alignment/factual consistency metric", [r"alignment function", r"factual consistency"]),
    "tang2024minicheck": ("MiniCheck: small fact-checkers approach GPT-4 at much lower cost; grounding documents", [r"GPT-4", r"400\s*[x×]|cheaper|lower cost", r"grounding documents?"]),
    "niu2024ragtruth": ("RAGTruth: human annotations of hallucinations in RAG outputs (word/span level)", [r"word[- ]level", r"span", r"manually annotated|human annot"]),
    "surerag2026": ("combines a verifier with revision or abstention in RAG", [r"verif", r"abstain|abstention|revis"]),
    "pave2026": ("combines a verifier with revision or abstention in RAG", [r"verif|support score", r"revis|abstain"]),
    "rhrag2026": ("combines a verifier with revision or abstention in RAG", [r"verif|NLI", r"revis|abstain"]),
    "r2vc2026": ("combines a verifier with revision or abstention in RAG", [r"verif", r"abstain|abstention|calibrat"]),
    "geifman2017selective": ("selective prediction: abstain to trade coverage for risk", [r"risk-coverage|coverage", r"reject option|abstain"]),
    "kamath2020selective": ("selective question answering: abstain", [r"selective question answering", r"abstain"]),
    "geifman2019aurc": ("AURC summarises the risk-coverage curve", [r"AURC", r"area under the risk-coverage"]),
    "angelopoulos2024crc": ("CRC bounds expected loss; threshold rule (n/(n+1))R_hat + B/(n+1) <= alpha", [r"expected value of any monotone loss|expected loss", r"n\s*/?\s*\(?n\s*\+\s*1\)?|n\+1"]),
    "angelopoulos2021ltt": ("LTT bounds P(risk > alpha) <= delta via multiple testing", [r"multiple hypothesis testing|multiple testing", r"1\s*[-−]\s*δ|delta"]),
    "yadkori2024conformalabstention": ("conformal abstention for LLMs with guarantees", [r"conformal", r"abstain|abstention"]),
    "mohri2024conformalfactuality": ("conformal factuality guarantees for LMs", [r"conformal factuality|conformal", r"back-off|correctness guarantee"]),
    "li2024traq": ("TRAQ: conformal prediction for RAG QA with guarantees", [r"conformal prediction", r"retrieval"]),
    "kang2024cragcert": ("C-RAG: certified generation risk for RAG", [r"certif", r"generation risk"]),
    "thakur2021beir": ("BEIR HotpotQA corpus = Wikipedia (2017) introductory paragraphs", [r"HotpotQA", r"Wikipedia"]),
    "manakul2023selfcheckgpt": ("SelfCheckGPT: consistency of sampled responses", [r"sampled responses|stochastically", r"consisten"]),
    "nogueira2019monobert": ("cross-encoder passage reranking with BERT", [r"re-?rank", r"BERT"]),
}


def main() -> None:
    meta = {r["key"]: r for r in csv.DictReader((ROOT / "paper" / "refs_verification.csv").open(encoding="utf-8"))}
    keys = sys.argv[1:] or list(CLAIMS)
    for k in keys:
        claim, pats = CLAIMS[k]
        aid = meta[k]["arxiv_id"].strip()
        f = ROOT / "literature" / "pdfs" / f"{aid}.txt"
        print(f"\n=== {k} ({aid}) — {claim}")
        if not f.exists():
            print("   NO FULL TEXT")
            continue
        t = re.sub(r"\s+", " ", f.read_text(encoding="utf-8"))
        for p in pats:
            ms = list(re.finditer(p, t, re.I))
            print(f"  /{p}/: {len(ms)} hits")
            for m in ms[:2]:
                print("     …" + t[max(0, m.start() - 140): m.end() + 140] + "…")


if __name__ == "__main__":
    main()
