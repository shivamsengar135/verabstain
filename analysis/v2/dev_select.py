"""v2 development (dev items 0-299): choose among candidate answers D (grounded draft / fallback), R (standard RAG on
the same evidence) and C (closed-book) with verifier support and confidence signals. Compares simple rules and a
logistic selector with 5-fold cross-validation on dev only. Usage: .venv/Scripts/python.exe analysis/v2/dev_select.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from ragverify import data as D  # noqa: E402
from score import score_example  # noqa: E402

S = json.loads((ROOT / "results" / "analysis" / "v2_dev_support.json").read_text())
lg = lambda p: float(np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))))  # noqa: E731
norm = lambda a: a.strip().lower()  # noqa: E731


def feats(e, k):
    c = e["cand"]; a = norm(c[k])
    agree = sum(norm(c[j]) == a for j in "DRC") - 1
    return [lg(e["sup"][k]), lg(e["c1"]) if k == "C" else 0.0, float(k == "C"), float(k == "R"), agree,
            float(norm(c["C"]) == a) * lg(e["c1"]), float(e["insufficient"])]


rows = {}
for ds, d in S.items():
    exs = {x["id"]: x for x in D.load(ds, "dev", 300)}
    rows[ds] = [(i, e, {k: score_example(exs[i], {"pred": e["cand"][k], "native_abstain": False})["correct"] for k in "DRC"}) for i, e in d.items()]


def rule_acc(ds, pick):
    return np.mean([y[pick(e)] for _, e, y in rows[ds]])


rules = {
    "D (v1 no M5)": lambda e: "D",
    "R": lambda e: "R",
    "max sup": lambda e: max("DRC", key=lambda k: e["sup"][k]),
    "max sup D/R": lambda e: max("DR", key=lambda k: e["sup"][k]),
    "D unless C more supported +.3": lambda e: "C" if e["sup"]["C"] > e["sup"]["D"] + 0.3 else "D",
}
for name, f in rules.items():
    print(f"{name:32s}", {ds: round(rule_acc(ds, f), 3) for ds in rows})

# logistic selector: P(candidate correct | features), pick argmax; trained on pooled QA dev with 5-fold CV over items
allr = [(ds, r) for ds in rows for r in rows[ds]]
kf = KFold(5, shuffle=True, random_state=0)
acc = {ds: [] for ds in rows}
for tr, te in kf.split(allr):
    X = [feats(allr[j][1][1], k) for j in tr for k in "DRC"]; y = [allr[j][1][2][k] for j in tr for k in "DRC"]
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    for j in te:
        ds, (_, e, yy) = allr[j]
        p = m.predict_proba([feats(e, k) for k in "DRC"])[:, 1]
        acc[ds].append(yy["DRC"[int(np.argmax(p))]])
print(f"{'logistic selector (5-fold CV)':32s}", {ds: round(np.mean(v), 3) for ds, v in acc.items()})
print(f"{'oracle':32s}", {ds: round(np.mean([max(y.values()) for _, _, y in rows[ds]]), 3) for ds in rows})

# --- selective prediction: confidence of the chosen answer (selector probability) vs v1's M6 score, dev CV ---
from ragverify.metrics import at_coverage, risk_coverage  # noqa: E402
aurc = lambda c, y: risk_coverage(c, y)[2]  # noqa: E731
halluc_at = lambda c, y, v: at_coverage(c, y, v)  # noqa: E731
v1 = {}
for ds in rows:
    o = {x["id"]: x for x in map(json.loads, (ROOT / "results" / f"devB2_ragverify_{ds}" / "outputs.jsonl").open(encoding="utf-8"))}
    v1[ds] = o
m6 = json.loads((ROOT / "analysis" / "m6_score_model.json").read_text())
L = m6["logistic"]
for ds in rows:
    ys, cs = [], []
    for i, e, yy in rows[ds]:
        o = v1[ds][i]; sc = o["scores"]
        x = [lg(sc["support"]), lg(sc["tokenprob"]), lg(sc["c1"]), float(o["native_abstain"])]
        cs.append(float(np.dot(L["coef"], x) + L["intercept"]))
        ys.append(score_example(D.load(ds, "dev", 300)[0] if False else {**{}, **{}}, {}) if False else None)
    exs = {x["id"]: x for x in D.load(ds, "dev", 300)}
    y = np.array([score_example(exs[i], v1[ds][i])["correct"] for i, _, _ in rows[ds]], float)
    print(ds, "v1 acc", round(y.mean(), 3), "AURC", round(aurc(np.array(cs), y), 3), "H@0.8", halluc_at(np.array(cs), y, 0.8))
conf = {ds: [] for ds in rows}; corr = {ds: [] for ds in rows}
for tr, te in kf.split(allr):
    X = [feats(allr[j][1][1], k) for j in tr for k in "DRC"]; y = [allr[j][1][2][k] for j in tr for k in "DRC"]
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    for j in te:
        ds, (_, e, yy) = allr[j]
        p = m.predict_proba([feats(e, k) for k in "DRC"])[:, 1]
        b = int(np.argmax(p)); conf[ds].append(p[b]); corr[ds].append(yy["DRC"[b]])
for ds in rows:
    c, y = np.array(conf[ds]), np.array(corr[ds], float)
    print(ds, "v2 acc", round(y.mean(), 3), "AURC", round(aurc(c, y), 3), "H@0.8", halluc_at(c, y, 0.8))
