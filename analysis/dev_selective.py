"""Dev-only selective-prediction readout from pilot runs: AURC and hallucination rate at coverage 0.8, per
method and dataset, using each method's pre-specified confidence score (PROTOCOL §8). Pilot/maximal-mode
outputs, so RAG-Verify numbers are for the maximal configuration. Usage: python analysis/dev_selective.py"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ragverify import data as D  # noqa: E402
from ragverify.metrics import at_coverage, risk_coverage  # noqa: E402
from score import score_example  # noqa: E402

SCORE = {"rag_selfcheck": "selfcheck", "selfrag": "selfrag_final", "ragverify": "support"}
rows = defaultdict(dict)
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "pilot"
for d in sorted((ROOT / "results").glob(f"{PREFIX}_*")):
    if not (d / "DONE").exists():
        continue
    job = json.loads((d / "meta.json").read_text())["job"]
    exs = {e["id"]: e for e in D.load(job["dataset"], "dev", job["n"])}
    outs = [json.loads(l) for l in (d / "outputs.jsonl").open(encoding="utf-8")]
    key = SCORE.get(job["method"], "tokenprob")
    sc = np.array([o["scores"].get(key, o["scores"].get("c1", 0.0)) if isinstance(o["scores"].get(key, 0.0), (int, float)) else 0.0 for o in outs], float)
    cor = np.array([score_example(exs[o["id"]], o)["correct"] for o in outs], float)
    _, _, aurc = risk_coverage(sc, cor)
    rows[job["method"]][job["dataset"]] = (cor.mean(), aurc, at_coverage(sc, cor, 0.8)["hallucination_rate"])
ds = ["popqa_lt", "triviaqa", "hotpotqa", "pubhealth"]
print(f"{'method':16s} " + " ".join(f"{d[:8]:>20s}" for d in ds) + "   mean acc / AURC / H@0.8")
for m, r in sorted(rows.items()):
    cells = [r.get(d) for d in ds]
    txt = " ".join(f"{c[0]:.2f}/{c[1]:.3f}/{c[2]:.2f}".rjust(20) if c else " " * 20 for c in cells)
    full = [c for c in cells if c]
    mean = tuple(np.mean([c[i] for c in full]) for i in range(3)) if len(full) == 4 else None
    print(f"{m:16s} {txt}   " + (f"{mean[0]:.3f}/{mean[1]:.3f}/{mean[2]:.3f}" if mean else "incomplete"))
