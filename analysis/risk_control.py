"""Risk-control evaluation (PROTOCOL §12): repeated random 50/50 calibration/evaluation splits of one run.

For alpha in {0.05, 0.10, 0.20} and seeds 0..99: CRC threshold on the calibration half for the overall
hallucination rate (answered & wrong / all), LTT threshold (delta = 0.1) for error among answered; report
realised risk / coverage on the evaluation half: mean, sd, fraction of splits with risk > alpha.
Scores: RAG-Verify uses the frozen logistic model (analysis/m6_score_model.json); others PROTOCOL §8 scores.
Usage: .venv/Scripts/python.exe analysis/risk_control.py RUN_ID [RUN_ID ...]  -> results/<run>/risk_control.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ragverify.metrics import crc_threshold, ltt_threshold  # noqa: E402

SCORE = {"rag_selfcheck": "selfcheck", "selfrag": "selfrag_final", "ragverify_v2": "v2conf"}  # v2: follow-up study


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def method_scores(method: str, rows: list[dict]) -> np.ndarray:
    if method == "ragverify":
        m = json.loads((ROOT / "analysis" / "m6_score_model.json").read_text())["logistic"]
        X = np.array([[lg(r["scores"].get("support", r["scores"].get("c1", 0.0))), lg(r["scores"].get("tokenprob", r["scores"].get("c1", 0.0))),
                       lg(r["scores"].get("c1", 0.0)), float(r["native_abstain"])] for r in rows])
        return 1 / (1 + np.exp(-(X @ np.array(m["coef"]) + m["intercept"])))
    key = SCORE.get(method, "tokenprob")
    return np.array([float(r["scores"].get(key, 0.0)) for r in rows])


def evaluate(run_id: str) -> dict:
    d = ROOT / "results" / run_id
    meta = json.loads((d / "meta.json").read_text())
    rows = [json.loads(l) for l in (d / "scores.jsonl").open(encoding="utf-8")]
    s = method_scores(meta["job"]["method"], rows)
    wrong = np.array([1 - r["correct"] for r in rows], float)
    n = len(rows)
    out = {"run_id": run_id, "n": n}
    for alpha in (0.05, 0.10, 0.20):
        res = {"crc": [], "ltt": []}
        for seed in range(100):
            perm = np.random.default_rng(seed).permutation(n)
            cal, ev = perm[: n // 2], perm[n // 2:]
            lam = crc_threshold(s[cal], wrong[cal], alpha)
            ans = s[ev] >= lam
            res["crc"].append((float((ans * wrong[ev]).mean()), float(ans.mean())))
            lam2 = ltt_threshold(s[cal], wrong[cal], alpha, delta=0.1)
            ans2 = s[ev] >= lam2
            sel = float(wrong[ev][ans2].mean()) if ans2.any() else 0.0
            res["ltt"].append((sel, float(ans2.mean())))
        summ = {}
        for k, v in res.items():
            r = np.array(v)
            summ[k] = {"risk_mean": float(r[:, 0].mean()), "risk_sd": float(r[:, 0].std()),
                       "frac_risk_gt_alpha": float((r[:, 0] > alpha).mean()), "coverage_mean": float(r[:, 1].mean())}
        out[str(alpha)] = summ
    (d / "risk_control.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    for rid in sys.argv[1:]:
        o = evaluate(rid)
        print(rid, {a: (round(o[a]["crc"]["risk_mean"], 3), round(o[a]["crc"]["coverage_mean"], 3)) for a in ("0.05", "0.1", "0.2")})
