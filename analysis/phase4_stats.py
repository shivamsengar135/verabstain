"""Phase-4 analysis (PROTOCOL §1 H2/H3, §10 ablations, §17 entry 2026-11-05 13:50 for the specification).

1. Ablations (PopQA-LT, HotpotQA; first 500 items of the main test order): each variant vs the main RAG-Verify
   configuration on the same 500 items. Accuracy, hallucination at coverage 0.8 and AURC (M6 frozen logistic score;
   a variant without verifier support, -M4, falls back to the draft confidence c1 inside the frozen model, as the code
   does for every run), paired-bootstrap 95% CI of variant - main, and mean cost per example.
   H2: main (external verifier) vs `selfverify` (backbone judges support): AURC and H@0.8 CIs, plus cost.
   H3: `plusM1g09` (gate on) vs main (gate off): H3 holds if the CI lower bound of accuracy(gated) - accuracy(main)
       is above -0.01 AND mean LLM calls and mean retrievals are both lower with the gate.
2. Secondary backbone (Llama-3.2-3B Q4, prefix sec3b, n=500): RAG-Verify vs each baseline run, same tests as
   analysis/main_stats.py (bootstrap CIs, McNemar / permutation / bootstrap p, Holm within dataset x metric).
3. Quantization check: Llama-3.2-3B Q8_0 (prefix quant3bq8, first 200) vs Q4_K_M (first 200 of the sec3b runs).
Only finished runs (results/<id>/DONE) are used; missing runs are listed as missing. Costs are attributed per example
including cached calls (identical cost accounting for every method).
Usage: .venv/Scripts/python.exe analysis/phase4_stats.py  -> results/analysis/phase4_stats.json, paper/generated/table_*.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))
from ragverify import data as D  # noqa: E402
from ragverify.metrics import holm, mcnemar_exact, paired_bootstrap, paired_permutation  # noqa: E402
from main_stats import DS_NAME, N_BOOT, answered_at, aurc_boot, load_run, metrics  # noqa: E402
from score import score_run  # noqa: E402
from method_name import METHOD_NAME  # noqa: E402

RES = ROOT / "results"
GEN = ROOT / "paper" / "generated"
VARIANTS = [("plusM1g09", "+M1 (gate, $\\tau$=0.9)"), ("noM2", "$-$M2"), ("noM3", "$-$M3"), ("noM4", "$-$M4"), ("noM5", "$-$M5"),
            ("selfverify", "Self-verification (M4 = backbone)"), ("hhem", "Verifier: HHEM-2.1-open"),
            ("minicheckdeb", "Verifier: MiniCheck-DeBERTa-L"), ("k3", "$k$=3"), ("k10", "$k$=10"), ("rounds1", "Rounds = 1")]
COST_KEYS = ("llm_calls", "prompt_tokens", "completion_tokens", "llm_seconds", "retrievals", "verifier_pairs")
SEC_METHODS = [("B1", "Closed-book", "closed_book"), ("B2", "RAG top-k", "rag_topk"), ("B3", "RAG + reranker", "rag_rerank"),
               ("B5", "CRAG", "crag"), ("B7", "FLARE", "flare"), ("B9", "CoVe", "cove"), ("A1", "B2 + token prob.", "rag_topk"),
               ("A2", "B2 + SelfCheck", "rag_selfcheck")]


def ready(rid: str) -> bool:
    d = RES / rid
    if not (d / "DONE").exists():
        return False
    if not (d / "scores.jsonl").exists() or (d / "scores.jsonl").stat().st_mtime < (d / "outputs.jsonl").stat().st_mtime:
        score_run(rid)
    return True


def cost(rid: str, keep: set) -> dict:
    rows = [json.loads(l) for l in (RES / rid / "scores.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["id"] in keep]
    return {k: float(np.mean([r["cost"].get(k, 0) for r in rows])) for k in COST_KEYS}


def compare(a, b, idx) -> dict:
    """a, b = (correct, scores) on identical items; differences a - b with CIs and p-values."""
    (ac, as_), (bc, bs) = a, b
    ah, bh = answered_at(as_, 0.8) * (1 - ac), answered_at(bs, 0.8) * (1 - bc)
    da = aurc_boot(as_, ac, idx) - aurc_boot(bs, bc, idx)
    ma, mb = metrics(ac, as_), metrics(bc, bs)
    return {"accuracy": {**paired_bootstrap(ac, bc), "p": mcnemar_exact(ac, bc)["p"]},
            "halluc@0.8": {**paired_bootstrap(ah, bh), "p": paired_permutation(ah, bh)},
            "aurc": {"diff": ma["aurc"] - mb["aurc"], "ci_low": float(np.percentile(da, 2.5)),
                     "ci_high": float(np.percentile(da, 97.5)), "n": len(ac),
                     "p": float(min(1.0, 2 * min((da <= 0).mean(), (da >= 0).mean())))}}


def ablations() -> dict:
    out = {}
    for ds in ("popqa_lt", "hotpotqa"):
        keep = {e["id"] for e in D.load(ds, "test", 500)}
        ref_id = f"main_ragverify_{ds}"
        _, ids, rc, rs = load_run(ref_id, "ragverify", keep)
        assert len(ids) == 500
        idx = np.random.default_rng(0).integers(0, len(ids), size=(N_BOOT, len(ids)))
        o = {"main": {"run": ref_id, **metrics(rc, rs), "cost": cost(ref_id, keep)}, "variants": {}, "missing": []}
        for v, _ in VARIANTS:
            rid = f"abl_ragverify_{v}_{ds}"
            if not ready(rid):
                o["missing"].append(rid)
                continue
            _, vids, vc, vs = load_run(rid, "ragverify", keep)
            assert vids == ids, rid
            o["variants"][v] = {"run": rid, **metrics(vc, vs), "cost": cost(rid, keep), "diff_vs_main": compare((vc, vs), (rc, rs), idx)}
        g = o["variants"].get("plusM1g09")
        if g:
            o["H3"] = {"acc_diff_ci_low": g["diff_vs_main"]["accuracy"]["ci_low"],
                       "fewer_llm_calls": g["cost"]["llm_calls"] < o["main"]["cost"]["llm_calls"],
                       "fewer_retrievals": g["cost"]["retrievals"] < o["main"]["cost"]["retrievals"]}
            o["H3"]["holds"] = o["H3"]["acc_diff_ci_low"] > -0.01 and o["H3"]["fewer_llm_calls"] and o["H3"]["fewer_retrievals"]
        sv = o["variants"].get("selfverify")
        if sv:  # H2: external verifier (main) minus self-verification
            d = sv["diff_vs_main"]
            o["H2"] = {k: {"diff": -d[k]["diff"], "ci_low": -d[k]["ci_high"], "ci_high": -d[k]["ci_low"], "p": d[k]["p"]}
                       for k in ("aurc", "halluc@0.8")}
            o["H2"]["cost_main"], o["H2"]["cost_selfverify"] = o["main"]["cost"], sv["cost"]
        out[ds] = o
    return out


def secondary() -> dict:
    out = {}
    for ds in ("popqa_lt", "hotpotqa", "pubhealth"):
        rv = f"sec3b_ragverify_{ds}"
        if not ready(rv):
            out[ds] = {"missing": [rv]}
            continue
        _, ids, rc, rs = load_run(rv, "ragverify")
        idx = np.random.default_rng(0).integers(0, len(ids), size=(N_BOOT, len(ids)))
        o = {"ragverify": {"run": rv, **metrics(rc, rs), "cost": cost(rv, set(ids))}, "baselines": {}, "missing": []}
        for bid, name, m in SEC_METHODS:
            rid = f"sec3b_{m}_{ds}"
            if not ready(rid):
                o["missing"].append(rid)
                continue
            _, bids, bc, bs = load_run(rid, m)
            assert bids == ids, rid
            o["baselines"][bid] = {"name": name, "run": rid, **metrics(bc, bs), "diff": compare((rc, rs), (bc, bs), idx)}
        for k in ("accuracy", "halluc@0.8", "aurc"):
            pv = {b: e["diff"][k]["p"] for b, e in o["baselines"].items()}
            for b, p in holm(pv).items():
                o["baselines"][b]["diff"][k]["p_holm"] = p
        out[ds] = o
    return out


def quant() -> dict:
    out = {}
    for ds in ("popqa_lt", "hotpotqa"):
        keep = {e["id"] for e in D.load(ds, "test", 200)}
        o = {}
        for m in ("closed_book", "rag_topk", "ragverify"):
            q8, q4 = f"quant3bq8_{m}_{ds}", f"sec3b_{m}_{ds}"
            if not (ready(q8) and ready(q4)):
                o[m] = {"missing": [r for r in (q8, q4) if not (RES / r / "DONE").exists()]}
                continue
            _, i8, c8, s8 = load_run(q8, m, keep)
            _, i4, c4, s4 = load_run(q4, m, keep)
            assert i8 == i4 and len(i8) == 200
            idx = np.random.default_rng(0).integers(0, 200, size=(N_BOOT, 200))
            o[m] = {"q8": metrics(c8, s8), "q4": metrics(c4, s4), "diff_q8_minus_q4": compare((c8, s8), (c4, s4), idx)}
        out[ds] = o
    return out


def ci(x: dict) -> str:
    return f"{x['diff']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]"


def tables(r: dict) -> None:
    L = ["% generated by analysis/phase4_stats.py (variant minus main RAG-Verify, first 500 items; 95% paired-bootstrap CI)",
         "\\begin{tabular}{@{}lllll@{}}", "\\toprule Dataset & Variant & $\\Delta$Acc & $\\Delta$H@0.8 & $\\Delta$AURC \\\\ \\midrule"]
    for ds, o in r["ablations"].items():
        L.append(f"{DS_NAME[ds]} & main (acc {o['main']['accuracy']:.3f}, H@0.8 {o['main']['halluc@0.8']:.3f}, "
                 f"AURC {o['main']['aurc']:.3f}) & & & \\\\")
        for v, name in VARIANTS:
            e = o["variants"].get(v)
            cells = [ci(e["diff_vs_main"][k]) for k in ("accuracy", "halluc@0.8", "aurc")] if e else ["not run"] * 3
            L.append(f" & {name} & " + " & ".join(cells) + " \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    (GEN / "table_ablations.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    C = ["% generated by analysis/phase4_stats.py: mean cost per example (cached calls attributed), first 500 items",
         "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule Dataset & Configuration & LLM calls & Prompt tok. & Compl. tok. & "
         "Retrievals & Verifier pairs \\\\ \\midrule"]
    for ds, o in r["ablations"].items():
        for name, e in [("main", o["main"])] + [(n, o["variants"][v]) for v, n in VARIANTS if v in o["variants"]]:
            c = e["cost"]
            C.append(f"{DS_NAME[ds]} & {name} & {c['llm_calls']:.2f} & {c['prompt_tokens']:.0f} & {c['completion_tokens']:.0f} & "
                     f"{c['retrievals']:.2f} & {c['verifier_pairs']:.2f} \\\\")
    C += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "table_ablation_cost.tex").write_text("\n".join(C) + "\n", encoding="utf-8")

    S = ["% generated by analysis/phase4_stats.py: Llama-3.2-3B (Q4_K_M), 500 items per dataset",
         "\\begin{tabular}{@{}ll" + "ccc" * 3 + "@{}}", "\\toprule", " & & " + " & ".join(
             f"\\multicolumn{{3}}{{c}}{{{DS_NAME[d]}}}" for d in ("popqa_lt", "hotpotqa", "pubhealth")) + " \\\\",
         " & Method & " + " & ".join(["Acc & H@0.8 & AURC"] * 3) + " \\\\ \\midrule"]
    for bid, name, _ in [("", METHOD_NAME, None)] + SEC_METHODS:
        cells = []
        for d in ("popqa_lt", "hotpotqa", "pubhealth"):
            o = r["secondary"].get(d, {})
            e = o.get("ragverify") if bid == "" else o.get("baselines", {}).get(bid)
            cells += [f"{e[k]:.3f}" for k in ("accuracy", "halluc@0.8", "aurc")] if e else ["--"] * 3
        S.append(f"{bid} & {name} & " + " & ".join(cells) + " \\\\")
    S += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "table_secondary.tex").write_text("\n".join(S) + "\n", encoding="utf-8")

    Q = ["% generated by analysis/phase4_stats.py: Llama-3.2-3B Q8_0 minus Q4_K_M, first 200 items (95% paired-bootstrap CI)",
         "\\begin{tabular}{@{}lllll@{}}", "\\toprule Dataset & Method & $\\Delta$Acc & $\\Delta$H@0.8 & $\\Delta$AURC \\\\ \\midrule"]
    for ds, o in r["quant"].items():
        for m, e in o.items():
            cells = [ci(e["diff_q8_minus_q4"][k]) for k in ("accuracy", "halluc@0.8", "aurc")] if "q8" in e else ["not run"] * 3
            Q.append(f"{DS_NAME[ds]} & {m.replace('_', ' ')} & " + " & ".join(cells) + " \\\\")
    Q += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "table_quant.tex").write_text("\n".join(Q) + "\n", encoding="utf-8")


if __name__ == "__main__":
    res = {"ablations": ablations(), "secondary": secondary(), "quant": quant()}
    (RES / "analysis").mkdir(exist_ok=True)
    (RES / "analysis" / "phase4_stats.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    tables(res)
    for ds, o in res["ablations"].items():
        print(ds, "variants done", len(o["variants"]), "missing", len(o["missing"]), "H2", "H2" in o, "H3", o.get("H3", {}).get("holds"))
    print("secondary", {d: len(o.get("baselines", {})) for d, o in res["secondary"].items()}, "quant", {d: [m for m, e in o.items() if "q8" in e] for d, o in res["quant"].items()})
