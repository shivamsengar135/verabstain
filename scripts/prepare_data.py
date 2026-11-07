"""Build processed example files and fixed splits (seed 0) for every dataset.

Outputs: data/processed/<name>_<split>.jsonl (gitignored) and data/splits/<name>_<split>.ids (committed;
order = evaluation order, jobs take the first N). Test pools are only *built* here; nothing is scored.
Example schema: id, dataset, split, task ('qa'|'claim'), question, answers (list), ctxs (released
pre-retrieved passages, optional), corpus (benchmark-provided passage pool, optional), meta.
Usage: .venv/Scripts/python.exe scripts/prepare_data.py
"""
from __future__ import annotations

import ast
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, SPL = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "data" / "splits"
SEED = 0


def write(name: str, split: str, exs: list[dict], shuffle: bool = True) -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    SPL.mkdir(parents=True, exist_ok=True)
    if shuffle:
        random.Random(SEED).shuffle(exs)
    with (PROC / f"{name}_{split}.jsonl").open("w", encoding="utf-8") as f:
        for e in exs:
            e.update({"dataset": name, "split": split})
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    (SPL / f"{name}_{split}.ids").write_text("\n".join(e["id"] for e in exs) + "\n", encoding="utf-8")
    print(f"{name}_{split}: {len(exs)}")


def ctxs(lst: list[dict]) -> list[dict]:
    return [{"id": f"dpr:{c.get('id')}", "title": c.get("title", ""), "text": " ".join(str(c.get("text", "")).split()),
             "score": float(c["score"]) if c.get("score") not in (None, "") else None} for c in lst]


def selfrag_file(name: str) -> list[dict]:
    return [json.loads(l) for l in (RAW / "selfrag_eval" / "eval_data" / f"{name}.jsonl").open(encoding="utf-8")]


def popqa() -> None:
    lt = selfrag_file("popqa_longtail_w_gs")
    write("popqa_lt", "test", [{"id": f"popqa:{r['id']}", "task": "qa", "question": r["question"], "answers": r["answers"],
                                "ctxs": ctxs(r["ctxs"]), "meta": {"prop": r.get("prop"), "pop": r.get("pop")}} for r in lt])
    lt_ids = {str(r["id"]) for r in lt}
    df = pd.read_csv(RAW / "popqa" / "test.tsv", sep="\t")
    dev = df[(df.s_pop >= 100) & (~df.id.astype(str).isin(lt_ids))].sample(n=500, random_state=SEED)
    write("popqa_lt", "dev", [{"id": f"popqadev:{r.id}", "task": "qa", "question": r.question,
                               "answers": json.loads(r.possible_answers), "meta": {"prop": r.prop, "pop": int(r.s_pop)}}
                              for r in dev.itertuples()], shuffle=False)


def triviaqa() -> None:
    te = selfrag_file("triviaqa_test_w_gs")
    write("triviaqa", "test", [{"id": f"tqa:{r['id']}", "task": "qa", "question": r["question"], "answers": r["answers"],
                                "ctxs": ctxs(r["ctxs"])} for r in te])
    test_q = {r["question"].strip().lower() for r in te}
    tr = pd.read_parquet(RAW / "triviaqa" / "unfiltered.nocontext" / "train-00000-of-00001.parquet", columns=["question", "question_id", "answer"])
    tr = tr[~tr.question.str.strip().str.lower().isin(test_q)].sample(n=500, random_state=SEED)
    write("triviaqa", "dev", [{"id": f"tqadev:{r.question_id}", "task": "qa", "question": r.question,
                               "answers": list(r.answer["aliases"])} for r in tr.itertuples()], shuffle=False)


def hotpotqa() -> None:
    va = pd.read_parquet(RAW / "hotpotqa" / "distractor" / "validation-00000-of-00001.parquet")
    exs = []
    for r in va.itertuples():
        paras = [{"title": t, "text": " ".join(" ".join(s).split())} for t, s in zip(r.context["title"], r.context["sentences"])]
        gold = set(r.supporting_facts["title"])
        exs.append({"id": f"hotpot:{r.id}", "task": "qa", "question": r.question, "answers": [r.answer],
                    "meta": {"type": r.type, "level": r.level, "gold_titles": sorted(gold),
                             "distractor_paras": [p for p in paras if p["title"] not in gold],
                             "gold_paras": [p for p in paras if p["title"] in gold]}})
    write("hotpotqa", "test", exs)
    tr = pd.read_parquet(RAW / "hotpotqa" / "distractor" / "train-00000-of-00002.parquet", columns=["id", "question", "answer", "type", "level"])
    tr = tr.sample(n=500, random_state=SEED)
    write("hotpotqa", "dev", [{"id": f"hotpotdev:{r.id}", "task": "qa", "question": r.question, "answers": [r.answer],
                               "meta": {"type": r.type, "level": r.level}} for r in tr.itertuples()], shuffle=False)


def pubhealth() -> None:
    te = selfrag_file("health_claims_processed")
    write("pubhealth", "test", [{"id": f"pub:{i}", "task": "claim", "question": r["claim"], "answers": r["answers"],
                                 "ctxs": ctxs(r["ctxs"]), "meta": {"label": r["label"]}} for i, r in enumerate(te)], shuffle=False)
    f = RAW / "pubhealth" / "pubhealth_source" / "validation" / "0000.parquet"
    meta = json.loads(pq.read_schema(f).metadata[b"huggingface"])
    names = meta["info"]["features"]["label"]["names"]
    df = pd.read_parquet(f)
    df["lab"] = df.label.map(lambda i: names[i])
    df = df[df.lab.isin(["true", "false"])]
    df = df.sample(n=min(500, len(df)), random_state=SEED)
    test_claims = {r["claim"].strip().lower() for r in te}
    df = df[~df.claim.str.strip().str.lower().isin(test_claims)]
    write("pubhealth", "dev", [{"id": f"pubdev:{r.claim_id}", "task": "claim", "question": r.claim, "answers": [r.lab],
                                "meta": {"label_names": names}} for r in df.itertuples()], shuffle=False)


def rgb_process(inst: dict, noise_rate: float, fname: str, passage_num: int = 5) -> list[str]:
    """Copy of RGB evalue.py processdata (non-integration branches), seeded as in their main loop."""
    random.seed(2333)
    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num
    if "_fact" in fname:
        indexs = list(range(len(inst["positive"])))
        selected = random.sample(indexs, min(len(indexs), pos_num))
        docs = [inst["positive_wrong"][i] for i in selected]
        if neg_num > 0:
            docs += inst["negative"][:neg_num]
    else:
        if noise_rate == 1:
            neg_num, pos_num = passage_num, 0
        elif neg_num > len(inst["negative"]):
            neg_num = len(inst["negative"])
            pos_num = passage_num - neg_num
        elif pos_num > len(inst["positive"]):
            pos_num = len(inst["positive"])
            neg_num = passage_num - pos_num
        docs = inst["positive"][:pos_num] + inst["negative"][:neg_num]
    random.shuffle(docs)
    return docs


def rgb() -> None:
    en = [json.loads(l) for l in (RAW / "rgb" / "en.json").open(encoding="utf-8")]
    exs = []
    for rate in (0.0, 0.4, 0.8, 1.0):
        for inst in en:
            docs = rgb_process(inst, rate, "en")
            pool = inst["positive"] + inst["negative"]
            exs.append({"id": f"rgb:n{rate}:{inst['id']}", "task": "qa", "question": inst["query"], "answers": inst["answer"],
                        "ctxs": [{"id": f"rgbdoc:{inst['id']}:{pool.index(d) if d in pool else -1}", "title": "", "text": d} for d in docs],
                        "corpus": [{"id": f"rgbdoc:{inst['id']}:{i}", "title": "", "text": d} for i, d in enumerate(pool)],
                        "meta": {"noise_rate": rate, "condition": "rejection" if rate == 1.0 else "noise"}})
    fact = [json.loads(l) for l in (RAW / "rgb" / "en_fact.json").open(encoding="utf-8")]
    for inst in fact:
        docs = rgb_process(inst, 0.0, "en_fact")
        exs.append({"id": f"rgb:fact:{inst['id']}", "task": "qa", "question": inst["query"], "answers": [inst["answer"]],
                    "ctxs": [{"id": f"rgbfact:{inst['id']}:{k}", "title": "", "text": d} for k, d in enumerate(docs)],
                    "corpus": [{"id": f"rgbfact:{inst['id']}:{k}", "title": "", "text": d} for k, d in enumerate(docs)],
                    "meta": {"condition": "counterfactual", "fakeanswer": inst["fakeanswer"]}})
    write("rgb", "test", exs, shuffle=False)


def ragtruth() -> None:
    src = {}
    for l in (RAW / "ragtruth" / "source_info.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        src[r["source_id"]] = r
    out = {"train": [], "test": []}
    for l in (RAW / "ragtruth" / "response.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        s = src[r["source_id"]]
        if s["task_type"] != "QA":
            continue
        info = s["source_info"]
        out[r["split"]].append({"id": f"rt:{r['id']}", "task": "verify", "question": info["question"], "doc": info["passages"],
                                "response": r["response"], "label": int(len(r["labels"]) > 0), "model": r["model"]})
    write("ragtruth_qa", "test", out["test"])
    tr = out["train"]
    random.Random(SEED).shuffle(tr)
    write("ragtruth_qa", "dev", tr[:600], shuffle=False)


if __name__ == "__main__":
    for fn in (popqa, triviaqa, hotpotqa, pubhealth, rgb, ragtruth):
        fn()
