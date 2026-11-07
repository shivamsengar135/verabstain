"""Dev-only choice of RAG-Verify's M6 confidence score (PROTOCOL §5/§8).

Candidates: (a) verifier support; (b) final-answer token probability; (c) logistic regression on
[logit support, logit token prob, logit closed-book confidence, insufficient-evidence flag] fitted on dev.
Selection criterion: mean AURC over the 4 dev datasets, with 5-fold cross-validation (stratified by dataset)
for (c). The chosen model is refit on all dev data and saved to analysis/m6_score_model.json.
Usage: .venv/Scripts/python.exe analysis/score_select.py [RUN_PREFIX=pilot] [N_DEV=100]
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ragverify import data as D  # noqa: E402
from ragverify.metrics import risk_coverage  # noqa: E402
from score import score_example  # noqa: E402


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


X, y, ds = [], [], []
for name in ["popqa_lt", "triviaqa", "hotpotqa", "pubhealth"]:
    PREFIX = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    NDEV = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    d = ROOT / "results" / f"{PREFIX}_ragverify_{name}"
    exs = {e["id"]: e for e in D.load(name, "dev", NDEV)}
    for line in (d / "outputs.jsonl").open(encoding="utf-8"):
        o = json.loads(line)
        s = o["scores"]
        X.append([lg(s.get("support", 0.0)), lg(s.get("tokenprob", 0.0)), lg(s.get("c1", 0.0)), float(o["native_abstain"])])
        y.append(score_example(exs[o["id"]], o)["correct"])
        ds.append(name)
X, y, ds = np.array(X), np.array(y, float), np.array(ds)


def mean_aurc(score):
    return float(np.mean([risk_coverage(score[ds == n], y[ds == n])[2] for n in np.unique(ds)]))


res = {"support": mean_aurc(X[:, 0]), "tokenprob": mean_aurc(X[:, 1]), "c1": mean_aurc(X[:, 2])}
cv = np.zeros(len(y))
strat = np.array([f"{a}|{int(b)}" for a, b in zip(ds, y)])
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, strat):
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr])
    cv[te] = m.predict_proba(X[te])[:, 1]
res["logistic_cv"] = mean_aurc(cv)
best = min(res, key=res.get)
full = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
out = {"mean_aurc_dev": res, "chosen": best, "n": int(len(y)),
       "logistic": {"features": ["logit support", "logit tokenprob", "logit c1", "insufficient"],
                    "coef": full.coef_[0].tolist(), "intercept": float(full.intercept_[0])}}
(ROOT / "analysis" / "m6_score_model.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out["mean_aurc_dev"]), "->", best)
from ragverify.metrics import at_coverage  # noqa: E402
for n in np.unique(ds):
    m = ds == n
    print(f"ragverify[{best}] {n}: acc {y[m].mean():.2f} AURC {risk_coverage(cv[m], y[m])[2]:.3f} "
          f"H@0.8 {at_coverage(cv[m], y[m], 0.8)['hallucination_rate']:.2f}")
