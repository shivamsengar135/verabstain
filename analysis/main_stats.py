"""Main test-set analysis (PROTOCOL §1 H1, §8 metrics, §9 statistics) on the primary backbone.

For each test dataset: accuracy (abstention off), hallucination rate at coverage 0.6/0.7/0.8/0.9 and AURC for
RAG-Verify and every baseline (B1-B10, A1, A2), each with its pre-specified confidence score (§8; RAG-Verify = frozen
logistic M6 score, analysis/m6_score_model.json). RAG-Verify - baseline differences: paired bootstrap 95% CI (10,000
resamples, seed 0) for all three primary metrics; McNemar exact test (accuracy), paired permutation test (10,000;
hallucination indicators at coverage 0.8), bootstrap two-sided p (AURC); Holm-Bonferroni over the 12 comparisons
within each dataset x primary metric. H1 holds on a dataset only if RAG-Verify's hallucination rate at coverage 0.8 is
significantly lower (Holm-adjusted p < 0.05, difference < 0) than that of every baseline.

Implementation notes (no protocol change):
  * A1 is B2's outputs scored by answer-token probability, which is also B2's own §8 confidence score, so A1 and B2
    have identical primary metrics; both rows are reported as the protocol lists them.
  * A2 (SelfCheck consistency) has 3 seeds: tests use seed 0; the table reports mean +- sd over seeds 0-2.
Usage: .venv/Scripts/python.exe analysis/main_stats.py
Writes results/analysis/main_stats.json and paper/generated/table_main.tex, table_main_tests.tex.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from ragverify.metrics import at_coverage, holm, mcnemar_exact, paired_bootstrap, paired_permutation, risk_coverage  # noqa: E402
from risk_control import method_scores  # noqa: E402
from method_name import METHOD_NAME  # noqa: E402

DATASETS = ["popqa_lt", "triviaqa", "hotpotqa", "pubhealth", "rgb"]
DS_NAME = {"popqa_lt": "PopQA-LT", "triviaqa": "TriviaQA", "hotpotqa": "HotpotQA", "pubhealth": "PubHealth", "rgb": "RGB"}
BASELINES = [("B1", "Closed-book", "closed_book"), ("B2", "RAG top-k", "rag_topk"), ("B3", "RAG + reranker", "rag_rerank"),
             ("B4", "RAG large-k", "rag_largek"), ("B5", "CRAG", "crag"), ("B6", "Chain-of-Note", "chain_of_note"),
             ("B7", "FLARE", "flare"), ("B8", "Self-RAG", "selfrag"), ("B9", "CoVe", "cove"), ("B10", "RARR", "rarr"),
             ("A1", "B2 + token-prob. abstention", "rag_topk"), ("A2", "B2 + SelfCheck", "rag_selfcheck")]
COVS = (0.6, 0.7, 0.8, 0.9)
N_BOOT, CHUNK = 10_000, 1000


def load_run(rid: str, method: str, keep: set | None = None):
    rows = [json.loads(l) for l in (ROOT / "results" / rid / "scores.jsonl").open(encoding="utf-8")]
    rows = sorted((r for r in rows if keep is None or r["id"] in keep), key=lambda r: r["id"])
    return rid, [r["id"] for r in rows], np.array([r["correct"] for r in rows], float), method_scores(method, rows)


def load(method: str, ds: str, seed: int = 0):
    return load_run(f"main_{method}{'' if seed == 0 else f'_s{seed}'}_{ds}", method)


def answered_at(scores: np.ndarray, cov: float) -> np.ndarray:
    n = len(scores)
    order = np.lexsort((np.arange(n), -scores))
    ans = np.zeros(n, bool)
    ans[order[: int(np.ceil(cov * n))]] = True
    return ans


def aurc_boot(scores: np.ndarray, correct: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """AURC on each bootstrap resample (rows of idx); ties broken by resample position, as in risk_coverage."""
    out = np.empty(len(idx))
    for s in range(0, len(idx), CHUNK):
        sc, er = scores[idx[s:s + CHUNK]], 1 - correct[idx[s:s + CHUNK]]
        order = np.argsort(-sc, axis=1, kind="stable")
        e = np.take_along_axis(er, order, axis=1)
        k = np.arange(1, e.shape[1] + 1)
        out[s:s + CHUNK] = (np.cumsum(e, axis=1) / k).mean(axis=1)
    return out


def metrics(correct, scores) -> dict:
    m = {"accuracy": float(correct.mean()), "aurc": risk_coverage(scores, correct)[2], "n": int(len(correct))}
    for c in COVS:
        m[f"halluc@{c}"] = at_coverage(scores, correct, c)["hallucination_rate"]
    return m


def main() -> None:
    res = {"datasets": {}}
    for ds in DATASETS:
        rv_id, ids, rv_c, rv_s = load("ragverify", ds)
        rv_h = (answered_at(rv_s, 0.8) * (1 - rv_c))
        idx = np.random.default_rng(0).integers(0, len(ids), size=(N_BOOT, len(ids)))
        rv_aurc_b = aurc_boot(rv_s, rv_c, idx)
        dsr = {"ragverify": {"run": rv_id, **metrics(rv_c, rv_s)}, "baselines": {}}
        p = {"accuracy": {}, "halluc@0.8": {}, "aurc": {}}
        for bid, name, method in BASELINES:
            b_id, b_ids, b_c, b_s = load(method, ds)
            assert b_ids == ids, f"item mismatch {b_id}"
            entry = {"id": bid, "name": name, "run": b_id, **metrics(b_c, b_s)}
            if method == "rag_selfcheck":
                seeds = [metrics(*load(method, ds, s)[2:]) for s in (0, 1, 2)]
                entry["seed_sd"] = {k: float(np.std([m[k] for m in seeds])) for k in seeds[0] if k != "n"}
                entry["seed_mean"] = {k: float(np.mean([m[k] for m in seeds])) for k in seeds[0] if k != "n"}
            b_h = answered_at(b_s, 0.8) * (1 - b_c)
            d_aurc = rv_aurc_b - aurc_boot(b_s, b_c, idx)
            entry["diff"] = {
                "accuracy": paired_bootstrap(rv_c, b_c),
                "halluc@0.8": paired_bootstrap(rv_h, b_h),
                "aurc": {"diff": dsr["ragverify"]["aurc"] - entry["aurc"], "ci_low": float(np.percentile(d_aurc, 2.5)),
                         "ci_high": float(np.percentile(d_aurc, 97.5)), "n": len(ids)},
            }
            p["accuracy"][bid] = mcnemar_exact(rv_c, b_c)["p"]
            p["halluc@0.8"][bid] = paired_permutation(rv_h, b_h)
            p["aurc"][bid] = float(min(1.0, 2 * min((d_aurc <= 0).mean(), (d_aurc >= 0).mean())))
            dsr["baselines"][bid] = entry
        for metric, pv in p.items():
            adj = holm(pv)
            for bid in pv:
                dsr["baselines"][bid]["diff"][metric].update({"p": pv[bid], "p_holm": adj[bid]})
        dsr["H1_holds"] = all(e["diff"]["halluc@0.8"]["diff"] < 0 and e["diff"]["halluc@0.8"]["p_holm"] < 0.05
                              for e in dsr["baselines"].values())
        res["datasets"][ds] = dsr
        print(ds, "H1", dsr["H1_holds"], flush=True)
    out = ROOT / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "main_stats.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    tables(res)


def f3(x):
    return f"{x:.3f}"


def tables(res: dict) -> None:
    gen = ROOT / "paper" / "generated"
    rows = [("", METHOD_NAME, "ragverify")] + [(b, n, b) for b, n, _ in BASELINES]
    L = ["% generated by analysis/main_stats.py from results/*/scores.jsonl (do not edit)",
         "\\begin{tabular}{@{}ll" + "ccc" * len(DATASETS) + "@{}}", "\\toprule",
         " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DS_NAME[d]}}}" for d in DATASETS) + " \\\\",
         " & Method & " + " & ".join(["Acc$\\uparrow$ & H@0.8$\\downarrow$ & AURC$\\downarrow$"] * len(DATASETS)) + " \\\\",
         "\\midrule"]
    for bid, name, key in rows:
        cells = []
        for d in DATASETS:
            r = res["datasets"][d]
            m = r["ragverify"] if key == "ragverify" else r["baselines"][key]
            cells += [f3(m["accuracy"]), f3(m["halluc@0.8"]), f3(m["aurc"])]
        L.append(f"{bid} & {name} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (gen / "table_main.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    T = ["% generated by analysis/main_stats.py: RAG-Verify minus baseline, 95% paired-bootstrap CI, Holm-adjusted p",
         "\\begin{longtable}{@{}lllll@{}}", "\\toprule Dataset & vs. & $\\Delta$Acc [CI] ($p_{Holm}$) & "
         "$\\Delta$H@0.8 [CI] ($p_{Holm}$) & $\\Delta$AURC [CI] ($p_{Holm}$) \\\\ \\midrule"]
    for d in DATASETS:
        for bid, e in res["datasets"][d]["baselines"].items():
            cells = []
            for k in ("accuracy", "halluc@0.8", "aurc"):
                x = e["diff"][k]
                cells.append(f"{x['diff']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}] ({x['p_holm']:.3g})")
            T.append(f"{DS_NAME[d]} & {bid} & " + " & ".join(cells) + " \\\\")
        T.append(f"\\multicolumn{{5}}{{l}}{{H1 on {DS_NAME[d]}: "
                 f"{'holds' if res['datasets'][d]['H1_holds'] else 'does not hold'}}} \\\\ \\midrule")
    T += ["\\bottomrule", "\\end{longtable}"]
    (gen / "table_main_tests.tex").write_text("\n".join(T) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
