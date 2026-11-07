"""Component-level verifier evaluation on RAGTruth QA (response-level hallucination detection).

Score of a response = min over its sentences of support(doc = source passages, claim = sentence).
Reports AUROC (hallucinated = positive, score = 1 - support), and F1 at the threshold that maximises F1 on
the *dev* split (applied unchanged to test). Usage:
  .venv/Scripts/python.exe analysis/verifier_ragtruth.py dev minicheck-ft5,minicheck-deb,hhem-2.1-open
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragverify import data as D  # noqa: E402
from ragverify.verifier import make, sent_split  # noqa: E402
sys.path.insert(0, str(ROOT / "analysis"))
from faithfulness import wait_cool  # noqa: E402

split, names = sys.argv[1], sys.argv[2].split(",")
exs = D.load("ragtruth_qa", split)
out_path = ROOT / "analysis" / f"verifier_ragtruth_{split}.json"
res = json.loads(out_path.read_text()) if out_path.exists() else {}
dev_thr = {}
if split == "test" and (ROOT / "analysis" / "verifier_ragtruth_dev.json").exists():
    dev_thr = {k: v["best_threshold"] for k, v in json.loads((ROOT / "analysis" / "verifier_ragtruth_dev.json").read_text()).items()}
y = np.array([e["label"] for e in exs])
for name in names:
    v = make(name, device="cuda", batch_size=8)
    pairs_d, pairs_c, owner = [], [], []
    for i, e in enumerate(exs):
        for s in sent_split(e["response"]):
            if len(s.split()) >= 3:
                pairs_d.append(e["doc"])
                pairs_c.append(s)
                owner.append(i)
    sc = []
    for s0 in range(0, len(pairs_c), 128):  # thermal wait between chunks (same limits as the runner's guard)
        wait_cool()
        sc += list(v.score(pairs_d[s0:s0 + 128], pairs_c[s0:s0 + 128]))
    resp = np.ones(len(exs))
    for i, s in zip(owner, sc):
        resp[i] = min(resp[i], s)
    auc = roc_auc_score(y, 1 - resp)
    ths = np.unique(resp)
    f1s = [f1_score(y, (resp < t).astype(int)) for t in ths]
    best_t = float(ths[int(np.argmax(f1s))])
    t_use = dev_thr.get(name, best_t)
    res[name] = {"auroc": float(auc), "best_threshold": best_t, "f1_at_dev_threshold": float(f1_score(y, (resp < t_use).astype(int))),
                 "threshold_used": t_use, "n": int(len(exs)), "pairs": len(sc), "positive_rate": float(y.mean())}
    print(name, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in res[name].items()}, flush=True)
    v.unload()
out_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
