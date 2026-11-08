"""Fit and freeze the two small models of RAG-Verify v2 on dev items 0-299 only (PROTOCOL §17 v2 entry).
QA selector: logistic P(candidate correct) on the features of analysis/v2/dev_select.py, pooled over PopQA-LT, TriviaQA
and HotpotQA dev (900 items x 3 candidates). Claim rule: logistic P(true) on logit MiniCheck support of the claim
(PubHealth dev, 300 items; devB2_ragverify traces). Output: analysis/v2/v2_model.json.
Usage: .venv/Scripts/python.exe analysis/v2/fit_v2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "v2")); sys.path.insert(0, str(ROOT / "src"))
import io, contextlib  # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    import dev_select as S  # noqa: E402  (reuses its rows/features; its printed CV results are suppressed here)
from ragverify import data as D  # noqa: E402

X = [S.feats(e, k) for ds in S.rows for _, e, _ in S.rows[ds] for k in "DRC"]
y = [yy[k] for ds in S.rows for _, _, yy in S.rows[ds] for k in "DRC"]
sel = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)

exs = {e["id"]: e for e in D.load("pubhealth", "dev", 300)}
g = [json.loads(l) for l in (ROOT / "results" / "devB2_ragverify_pubhealth" / "outputs.jsonl").open(encoding="utf-8")]
xs = np.array([[S.lg(o["trace"]["s_claim"])] for o in g]); ys = np.array([exs[o["id"]]["answers"][0] == "true" for o in g], int)
clm = LogisticRegression(C=1.0).fit(xs, ys)
out = {"selector": {"features": ["logit support", "logit c1 (closed-book cand.)", "is C", "is R", "n other cands agreeing",
                                 "agrees with C * logit c1", "draft insufficient"],
                    "coef": sel.coef_[0].tolist(), "intercept": float(sel.intercept_[0]), "n_rows": len(y)},
       "claim": {"features": ["logit MiniCheck support of the claim"], "coef": clm.coef_[0].tolist(),
                 "intercept": float(clm.intercept_[0]), "n": len(ys), "dev_acc_in_sample": float((clm.predict(xs) == ys).mean())}}
(ROOT / "analysis" / "v2" / "v2_model.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
