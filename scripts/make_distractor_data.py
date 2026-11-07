"""HotpotQA distractor robustness data (PROTOCOL §11, specification in §17 entry 2026-11-05 15:10).

For the first 300 items of the fixed HotpotQA test order that have >= 4 distractor paragraphs (3 of the first 303
have fewer; selection by data availability only): base = the 5 passages B2 (RAG top-k) received in the main run
(hybrid static list, from results/main_rag_topk_hotpotqa/outputs.jsonl trace.pids). For p in {0, 25, 50, 75}%,
m = floor(p/100 * 5 + 0.5) = 0/1/3/4 of the lowest-ranked base passages are replaced by the first m HotpotQA distractor
paragraphs of that question. Each example gets ctxs = these 5 passages and corpus = the same 5 (so every method,
including reranking and follow-up search, sees only them; same design as RGB). Only passage ids from the B2 trace and
HotpotQA's distractor paragraphs are used; no prediction or correctness of any run is read.
Usage: .venv/Scripts/python.exe scripts/make_distractor_data.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

N, K = 300, 5
LEVELS = {0: 0, 25: 1, 50: 3, 75: 4}

off = np.load(ROOT / "indexes" / "wiki17" / "offsets.npy", mmap_mode="r")
mm = np.memmap(ROOT / "indexes" / "wiki17" / "passages.bin", dtype=np.uint8, mode="r")


def passage(pid: str) -> dict:
    row = int(pid.split(":", 1)[1])
    t, x = bytes(mm[off[row]:off[row + 1]]).decode("utf-8").split("\t", 1)
    return {"id": pid, "title": t, "text": x}


full = [l for l in (ROOT / "data" / "splits" / "hotpotqa_test.ids").read_text(encoding="utf-8").splitlines() if l.strip()][:1000]
exs = {}
with (ROOT / "data" / "processed" / "hotpotqa_test.jsonl").open(encoding="utf-8") as f:
    for line in f:
        e = json.loads(line)
        exs[e["id"]] = e
order = [i for i in full if len(exs[i]["meta"]["distractor_paras"]) >= max(LEVELS.values())][:N]
pids = {}
with (ROOT / "results" / "main_rag_topk_hotpotqa" / "outputs.jsonl").open(encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        pids[o["id"]] = o["trace"]["pids"][:K]
missing = [i for i in order if i not in pids or len(pids[i]) < K]
assert not missing, f"{len(missing)} items lack a 5-passage B2 list"

for p, m in LEVELS.items():
    name = f"hotpotqa_dist{p:02d}"
    with (ROOT / "data" / "processed" / f"{name}_test.jsonl").open("w", encoding="utf-8") as out:
        for i in order:
            e = exs[i]
            base = [passage(pid) for pid in pids[i]]
            dis = [{"id": f"hpdis:{i}:{j}", "title": d["title"], "text": d["text"]} for j, d in enumerate(e["meta"]["distractor_paras"][:m])]
            assert len(dis) == m
            ctxs = base[: K - m] + dis
            rec = {"id": i, "dataset": name, "split": "test", "task": e["task"], "question": e["question"], "answers": e["answers"],
                   "ctxs": ctxs, "corpus": ctxs, "meta": {"distractor_level": p, "n_replaced": m, "base_pids": pids[i]}}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (ROOT / "data" / "splits" / f"{name}_test.ids").write_text("\n".join(order) + "\n", encoding="utf-8")
    print(name, len(order), "items, replaced", m)
