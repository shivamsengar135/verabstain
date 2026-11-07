"""All prompt templates (versioned). Changing any template changes PROMPT_VERSION and therefore the
LLM cache keys. Sources are noted per template; "adapted" means re-worded for a chat model and
short-answer QA, which is recorded as an adaptation in PROTOCOL.md.
"""
from __future__ import annotations

PROMPT_VERSION = "p1"

SYSTEM = "You are a careful assistant for factual question answering."

# ---- task framing shared by every method ----
QA_ANSWER_RULE = "Give only the short answer (a name, number, date or short phrase), without explanation."
CLAIM_INST = "Is the following statement correct or not? Say true if it's correct; otherwise say false."  # Self-RAG TASK_INST["fever"]


def fmt_passages(passages: list[dict], start: int = 1) -> str:
    return "\n\n".join(f"[{i}] {p['title']}: {p['text']}" for i, p in enumerate(passages, start))


def task_line(ex: dict) -> str:
    if ex["task"] == "claim":
        return f"{CLAIM_INST}\nStatement: {ex['question']}"
    return f"Question: {ex['question']}"


def answer_rule(ex: dict) -> str:
    return "Answer with exactly one word: true or false." if ex["task"] == "claim" else QA_ANSWER_RULE


# ---- closed book (no retrieval baseline; also RAG-Verify M1 draft: identical payload -> shared cache) ----
def closed_book(ex: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{task_line(ex)}\n\n{answer_rule(ex)}"}]


# ---- standard RAG (top-k / reranked / large-k baselines; abstention baselines reuse it) ----
def standard_rag(ex: dict, passages: list[dict]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Passages:\n{fmt_passages(passages)}\n\n{task_line(ex)}\n\n"
                                        f"Use the passages to answer. {answer_rule(ex)}"}]


# ---- RAG-Verify M3: evidence-grounded drafting with citations and an explicit abstain option ----
INSUFFICIENT = "insufficient evidence"


def grounded_draft(ex: dict, passages: list[dict]) -> list[dict]:
    if ex["task"] == "claim":
        fmt = ("Reply in exactly two lines:\nAnswer: true or false\n"
               "Evidence: one sentence from your reading of the passages that justifies the answer, with citations like [2].")
    else:
        fmt = ("Reply in exactly two lines:\nAnswer: <short answer only>\n"
               "Statement: <one complete sentence that states the answer to the question, with citations like [2]>")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Passages:\n{fmt_passages(passages)}\n\n{task_line(ex)}\n\n"
                                        f"Answer using only information in the passages. If the passages do not contain "
                                        f"the information needed, reply exactly: Answer: {INSUFFICIENT}\n{fmt}"}]


# ---- RAG-Verify M5: focused query and minimal revision ----
def focused_query(ex: dict, statement: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{task_line(ex)}\nCurrent answer statement (not supported by the evidence so far): {statement}\n\n"
                                        "Write one short search query (at most 12 words) that would find a Wikipedia passage "
                                        "confirming or correcting this statement. Reply with the query only."}]


def revise(ex: dict, passages: list[dict], old_answer: str) -> list[dict]:
    msgs = grounded_draft(ex, passages)
    msgs[1]["content"] += (f"\n\nA previous answer was '{old_answer}', but it was not supported by its evidence. "
                           "Keep it only if the passages above support it; otherwise correct it.")
    return msgs


# ---- ablation: LLM self-verification instead of the external verifier ----
def self_verify(statement: str, passages: list[dict]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Passages:\n{fmt_passages(passages)}\n\nClaim: {statement}\n\n"
                                        "Is the claim fully supported by the passages? Answer yes or no."}]


# ---- CRAG (Yan et al., 2024): generator prompt from scripts/CRAG_Inference.py (format_prompt, 'llama'
#      branch), keyword extraction prompts from scripts/utils.py (extract_keywords) ----
def crag_generate(ex: dict, knowledge: str) -> list[dict]:
    if ex["task"] == "claim":
        body = ("Read the documents and answer the question: Is the following statement correct or not? \n\nDocuments: "
                + knowledge + "\n\nStatement: " + ex["question"] + "\n\nOnly say true if the statement is true; otherwise say false.")
    else:
        body = ("Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: " + knowledge
                + "\n\nInstruction: Answer the question: " + ex["question"])
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body + "\n\n" + answer_rule(ex)}]


CRAG_KW_POPQA = ("Extract at most three keywords separated by comma from the following dialogues and questions as queries for the web search, "
                 "including topic background within dialogues and main intent within questions. \n\nquestion: What is Henry Feilden's occupation?\n"
                 "query: Henry Feilden, occupation\n\nquestion: In what city was Billy Carlson born?\nquery: city, Billy Carlson, born\n\n"
                 "question: What is the religion of John Gwynn?\nquery: religion of John Gwynn\n\nquestion: What sport does Kiribati men's national "
                 "basketball team play?\nquery: sport, Kiribati men's national basketball team play\n\nquestion: {question}\nquery: ")
CRAG_KW_PUBQA = ("Extract at most three keywords separated by comma from the claim as queries to extract the key information. \n\nclaim: A mother "
                 "revealed to her child in a letter after her death that she had just one eye because she had donated the other to him.\nquery: a mother "
                 "had one eye, donated the other eye to her child after her death.\n\nclaim: WWE wrestler Ric Flair was declared brain dead on 16 May 2019. "
                 "\nquery: WWE wrestler Ric Flair, brain dead on 16 May 2019\n\nclaim: Current expenditures could likely cover the estimated costs of "
                 "Medicare for All.\nquery: Current expenditures could likely cover the estimated costs of Medicare for All.\n\nclaim: Measles outbreak "
                 "kills more than 1,200 in Madagascar.\nquery: Measles outbreak kills more than 1,200 in Madagascar.\n\nclaim: {question}\nquery:  ")


def crag_keywords(ex: dict) -> list[dict]:
    tpl = CRAG_KW_PUBQA if ex["task"] == "claim" else CRAG_KW_POPQA
    return [{"role": "user", "content": tpl.format(question=ex["question"])}]


# ---- Chain-of-Note (Yu et al., 2024): reading notes per passage, then answer (prompting adaptation) ----
def chain_of_note(ex: dict, passages: list[dict]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Passages:\n{fmt_passages(passages)}\n\n{task_line(ex)}\n\n"
                                        "First write a brief reading note for each passage ([1], [2], ...): say whether it is relevant to the "
                                        "question and what information it provides. Then decide: if the passages contain the answer, answer from "
                                        "them; if they are relevant but incomplete, combine them with what you know; if none is relevant, "
                                        "answer from your own knowledge, or answer 'unknown' if you do not know.\n"
                                        "Format: one line per note ('Note [1]: ...'), then a last line 'Answer: <answer>', where <answer> is "
                                        + ("true or false." if ex["task"] == "claim" else "only a short answer (a name, number, date or short phrase).")}]


# ---- FLARE (Jiang et al., 2023): retrieval-conditioned generation of the answer sentence ----
def flare_generate(ex: dict, passages: list[dict], prefix: str = "") -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": (f"Passages:\n{fmt_passages(passages)}\n\n" if passages else "")
             + f"{task_line(ex)}\n\nAnswer in one short sentence that contains the answer."
             + (" Start the sentence with true or false." if ex["task"] == "claim" else "")}]
    if prefix:
        msgs.append({"role": "assistant", "content": prefix})
    return msgs


def extract_short(ex: dict, sentence: str) -> list[dict]:
    """Shared post-processing call for sentence-style outputs (FLARE, CoVe final): extract the short answer."""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{task_line(ex)}\nResponse: {sentence}\n\nWhat is the answer given in the response? {answer_rule(ex)}"}]


# ---- Chain-of-Verification, factored variant (Dhuliawala et al., 2024), adapted to RAG ----
def cove_plan(ex: dict, baseline: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{task_line(ex)}\nBaseline answer: {baseline}\n\n"
                                        "Write up to three short verification questions that would check the facts in the baseline answer. "
                                        "Each question must be answerable on its own (do not refer to 'the answer'). One question per line."}]


def cove_execute(question: str, passages: list[dict]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Passages:\n{fmt_passages(passages)}\n\nQuestion: {question}\n\n"
                                        "Answer the question briefly using the passages (one short sentence)."}]


def cove_final(ex: dict, baseline: str, qa: list[tuple[str, str]]) -> list[dict]:
    lines = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{task_line(ex)}\nBaseline answer: {baseline}\n\nVerification questions and answers:\n{lines}\n\n"
                                        "Using the verification answers, give the final answer, correcting the baseline answer if it is "
                                        f"inconsistent with them. {answer_rule(ex)}"}]
