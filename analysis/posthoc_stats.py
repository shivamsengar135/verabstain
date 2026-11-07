"""Post-hoc analyses P1 and P2 (PROTOCOL §17 entry of 2026-11-07, logged before any result existed). Not pre-registered;
reported separately from the main results.

P1  Equal evidence: Chain-of-Note and CRAG re-run on RAG-Verify's M2 evidence (runs posthoc_<m>_m2_<ds>, first 500 test
    items). RAG-Verify (same 500 items of its main run) minus each re-run baseline, and each re-run minus its original
    static-evidence run on the same items. Paired bootstrap 95% CI (10,000, seed 0); McNemar (accuracy) and paired
    permutation (H@0.8) p values, Holm over the 2 methods within dataset x metric (RAG-Verify comparisons only).
P2  Equal score fitting: every method's confidence score is replaced by a logistic model on all per-example signals the
    method produces (numeric fields of `scores`, logit-transformed when in (0,1), plus the native-abstention flag),
    two-fold cross-fitted on its test outputs (random 50/50 split, seed 0). AURC and H@0.8 are recomputed; RAG-Verify minus
    each baseline with paired bootstrap CIs (Holm over the 12 baselines per dataset x metric, as in the main analysis).
Outputs: results/analysis/posthoc_stats.json, paper/generated/table_posthoc_evidence.tex, table_posthoc_scores.tex.
Usage: .venv/Scripts/python.exe analysis/posthoc_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from ragverify.metrics import holm, mcnemar_exact, paired_bootstrap, paired_permutation  # noqa: E402
from main_stats import BASELINES, DATASETS, DS_NAME, answered_at, aurc_boot, load_run  # noqa: E402
from method_name import METHOD_NAME  # noqa: E402

N_BOOT = 10_000
P1_DS = ["popqa_lt", "triviaqa", "hotpotqa", "pubhealth"]
P1_M = [("B5", "CRAG", "crag"), ("B6", "Chain-of-Note", "chain_of_note")]


def ci(d: dict) -> str:
    f = lambda x: f"{x:+.3f}".replace("-", "$-$")  # noqa: E731
    return f"{f(d['diff'])} [{f(d['ci_low'])}, {f(d['ci_high'])}]"


def aurc(scores, correct) -> float:
    order = np.lexsort((np.arange(len(scores)), -scores))
    e = 1 - correct[order]
    return float((np.cumsum(e) / np.arange(1, len(e) + 1)).mean())


def boot_aurc_diff(sa, ca, sb, cb) -> dict:
    idx = np.random.default_rng(0).integers(0, len(ca), size=(N_BOOT, len(ca)))
    d = aurc_boot(sa, ca, idx) - aurc_boot(sb, cb, idx)
    return {"diff": aurc(sa, ca) - aurc(sb, cb), "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}


def compare(a, b) -> dict:
    """a, b = (correct, scores) on the same items (RAG-Verify first)."""
    (ca, sa), (cb, sb) = a, b
    ha, hb = answered_at(sa, 0.8) * (1 - ca), answered_at(sb, 0.8) * (1 - cb)
    acc = paired_bootstrap(ca, cb) | {"p": mcnemar_exact(ca, cb)["p"]}
    hal = paired_bootstrap(ha, hb) | {"p": paired_permutation(ha, hb)}
    return {"accuracy": acc, "halluc@0.8": hal, "aurc": boot_aurc_diff(sa, ca, sb, cb)}


def p1() -> dict:
    out = {}
    for d in P1_DS:
        _, ids, rc, rs = load_run(f"main_ragverify_{d}", "ragverify")
        keep = set(json.loads(l)["id"] for l in (ROOT / "results" / f"posthoc_crag_m2_{d}" / "scores.jsonl").open(encoding="utf-8"))
        _, ids, rc, rs = load_run(f"main_ragverify_{d}", "ragverify", keep)
        res = {"n": len(ids), "ragverify": {"accuracy": float(rc.mean()), "aurc": aurc(rs, rc),
                                            "halluc@0.8": float((answered_at(rs, 0.8) * (1 - rc)).mean())}}
        for bid, name, m in P1_M:
            _, i2, c2, s2 = load_run(f"posthoc_{m}_m2_{d}", m, keep)
            _, i3, c3, s3 = load_run(f"main_{m}_{d}", m, keep)
            assert ids == i2 == i3
            res[bid] = {"name": name, "m2": {"accuracy": float(c2.mean()), "aurc": aurc(s2, c2),
                                             "halluc@0.8": float((answered_at(s2, 0.8) * (1 - c2)).mean())},
                        "static": {"accuracy": float(c3.mean()), "aurc": aurc(s3, c3),
                                   "halluc@0.8": float((answered_at(s3, 0.8) * (1 - c3)).mean())},
                        "rv_minus_m2": compare((rc, rs), (c2, s2)), "m2_minus_static": compare((c2, s2), (c3, s3))}
        for k in ("accuracy", "halluc@0.8", "aurc"):
            adj = holm({b: res[b]["rv_minus_m2"][k]["p"] for b, _, _ in P1_M})
            for b, _, _ in P1_M:
                res[b]["rv_minus_m2"][k]["p_holm"] = adj[b]
        out[d] = res
        print("P1", d, "done", flush=True)
    return out


def feats(rows: list[dict]) -> np.ndarray:
    keys = sorted({k for r in rows for k, v in r["scores"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    cols = []
    for k in keys:
        v = np.array([float(r["scores"].get(k, np.nan)) for r in rows])
        v = np.where(np.isnan(v), np.nanmedian(v) if np.isfinite(np.nanmedian(v)) else 0.0, v)
        if v.min() >= 0 and v.max() <= 1:
            v = np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
        cols.append(v)
    cols.append(np.array([float(r.get("native_abstain", False)) for r in rows]))
    X = np.column_stack(cols)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def crossfit(rows: list[dict]) -> np.ndarray:
    X, y = feats(rows), np.array([r["correct"] for r in rows])
    perm = np.random.default_rng(0).permutation(len(y))
    halves = [perm[: len(y) // 2], perm[len(y) // 2:]]
    s = np.zeros(len(y))
    for a, b in ((0, 1), (1, 0)):
        tr, te = halves[a], halves[b]
        if len(set(y[tr])) < 2:
            s[te] = y[tr].mean()
            continue
        s[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return s


def rows_of(rid: str) -> list[dict]:
    return sorted((json.loads(l) for l in (ROOT / "results" / rid / "scores.jsonl").open(encoding="utf-8")), key=lambda r: r["id"])


def p2() -> dict:
    out = {}
    for d in DATASETS:
        rv = rows_of(f"main_ragverify_{d}")
        rc, rs = np.array([r["correct"] for r in rv], float), crossfit(rv)
        res = {"ragverify": {"aurc": aurc(rs, rc), "halluc@0.8": float((answered_at(rs, 0.8) * (1 - rc)).mean())}, "baselines": {}}
        for b, name, m in BASELINES:
            rid = f"main_{'rag_topk' if b == 'A1' else m}_{d}"
            br = rows_of(rid)
            assert [r["id"] for r in br] == [r["id"] for r in rv]
            bc, bs = np.array([r["correct"] for r in br], float), crossfit(br)
            c = compare((rc, rs), (bc, bs))
            res["baselines"][b] = {"name": name, "aurc": aurc(bs, bc), "halluc@0.8": float((answered_at(bs, 0.8) * (1 - bc)).mean()),
                                   "diff": {k: c[k] for k in ("halluc@0.8", "aurc")}}
        for k in ("halluc@0.8", "aurc"):
            bids = list(res["baselines"])
            adj = holm({b: res["baselines"][b]["diff"][k]["p"] for b in bids})
            for b in bids:
                res["baselines"][b]["diff"][k]["p_holm"] = adj[b]
        out[d] = res
        print("P2", d, "done", flush=True)
    return out


def tables(r1: dict, r2: dict) -> None:
    G = ROOT / "paper" / "generated"
    L = ["% generated by analysis/posthoc_stats.py (post-hoc P1): first 500 test items; " + METHOD_NAME + " minus baseline re-run on "
         "the same M2 evidence, and re-run minus original static-evidence run; 95% paired-bootstrap CI; * Holm p < 0.05",
         "\\begin{tabular}{@{}lllll@{}}", "\\toprule Dataset & Baseline & Comparison & $\\Delta$Acc & $\\Delta$H@0.8 \\\\ \\midrule"]
    for d, res in r1.items():
        for i, (b, name, _) in enumerate(P1_M):
            e = res[b]
            star = lambda x: "$^{*}$" if x.get("p_holm", 1) < 0.05 else ""  # noqa: E731
            L.append(f"{DS_NAME[d] if i == 0 else ''} & {b} {name} & {METHOD_NAME} $-$ {b}(M2) & {ci(e['rv_minus_m2']['accuracy'])}{star(e['rv_minus_m2']['accuracy'])} & "
                     f"{ci(e['rv_minus_m2']['halluc@0.8'])}{star(e['rv_minus_m2']['halluc@0.8'])} \\\\")
            L.append(f" & & {b}(M2) $-$ {b}(static) & {ci(e['m2_minus_static']['accuracy'])} & {ci(e['m2_minus_static']['halluc@0.8'])} \\\\")
        L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    (G / "table_posthoc_evidence.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    L = ["% generated by analysis/posthoc_stats.py (post-hoc P2): every method's score refitted by the same cross-fitted logistic "
         "recipe; H@0.8 / AURC; number of baselines " + METHOD_NAME + " is significantly better than (Holm p < 0.05, of 12)",
         "\\begin{tabular}{@{}lccccc@{}}", "\\toprule & " + " & ".join(DS_NAME[d] for d in DATASETS) + " \\\\ \\midrule"]
    L.append(f"{METHOD_NAME} H@0.8 / AURC & " + " & ".join(f"{r2[d]['ragverify']['halluc@0.8']:.3f} / {r2[d]['ragverify']['aurc']:.3f}" for d in DATASETS) + " \\\\")
    best = []
    for d in DATASETS:
        bb = min(r2[d]["baselines"].values(), key=lambda e: e["aurc"])
        best.append(f"{bb['halluc@0.8']:.3f} / {bb['aurc']:.3f}")
    L.append("Best baseline (by AURC) & " + " & ".join(best) + " \\\\")
    for k, nm in (("halluc@0.8", "H@0.8"), ("aurc", "AURC")):
        cells = []
        for d in DATASETS:
            bs = r2[d]["baselines"].values()
            good = sum(e["diff"][k]["p_holm"] < 0.05 and e["diff"][k]["diff"] < 0 for e in bs)
            bad = sum(e["diff"][k]["p_holm"] < 0.05 and e["diff"][k]["diff"] > 0 for e in bs)
            cells.append(f"{good} better / {bad} worse")
        L.append(f"{nm}: significant vs.\\ 12 & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (G / "table_posthoc_scores.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    r2 = p2()
    r1 = p1()
    (ROOT / "results" / "analysis" / "posthoc_stats.json").write_text(json.dumps({"P1": r1, "P2": r2}, indent=1), encoding="utf-8")
    tables(r1, r2)
    print("ok")


if __name__ == "__main__":
    main()
