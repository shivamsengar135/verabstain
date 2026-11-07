"""Generate RESULTS_SUMMARY.md from the analysis outputs (no hand-typed numbers; Rule 2).
Inputs: results/analysis/{main_stats,phase4_stats,distractor_stats,faithfulness}.json, results/main_ragverify_*/risk_control.json,
analysis/verifier_ragtruth_test.json, results/main_*/summary.json.
Usage: .venv/Scripts/python.exe analysis/make_results_summary.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from method_name import METHOD_NAME  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results" / "analysis"
DS = {"popqa_lt": "PopQA-LT", "triviaqa": "TriviaQA", "hotpotqa": "HotpotQA", "pubhealth": "PubHealth", "rgb": "RGB"}


def j(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ci(x: dict, pct: bool = False) -> str:
    f = 100 if pct else 1
    return f"{x['diff'] * f:+.3f} [{x['ci_low'] * f:+.3f}, {x['ci_high'] * f:+.3f}]"


def main() -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    L = [f"# RESULTS_SUMMARY (generated {datetime.now():%Y-%m-%d %H:%M} by analysis/make_results_summary.py, git {commit})", "",
         "All numbers below are read from results/ by script. Primary backbone: Qwen2.5-7B-Instruct Q4_K_M. "
         "Acc = accuracy (abstention off); H@0.8 = hallucination rate (answered & wrong / all) at coverage 0.8; AURC = area under "
         "the risk-coverage curve (lower is better). Differences are RAG-Verify minus baseline with 95% paired-bootstrap CIs; "
         "p values are Holm-adjusted over the 12 baselines within dataset x metric (PROTOCOL §9).", ""]

    ms = j(A / "main_stats.json")
    if ms:
        L += ["## H1 — main test sets (primary backbone)", ""]
        for d, r in ms["datasets"].items():
            rv = r["ragverify"]
            L.append(f"**{DS[d]}** (n = {rv['n']}): {METHOD_NAME} Acc {rv['accuracy']:.3f}, H@0.8 {rv['halluc@0.8']:.3f}, AURC {rv['aurc']:.3f}. "
                     f"H1 {'**holds**' if r['H1_holds'] else '**does not hold**'}.")
            for k, name in (("accuracy", "Acc"), ("halluc@0.8", "H@0.8"), ("aurc", "AURC")):
                good = [b for b, e in r["baselines"].items() if e["diff"][k]["p_holm"] < 0.05 and ((e["diff"][k]["diff"] > 0) == (k == "accuracy"))]
                bad = [b for b, e in r["baselines"].items() if e["diff"][k]["p_holm"] < 0.05 and ((e["diff"][k]["diff"] > 0) != (k == "accuracy"))]
                best = (max if k == "accuracy" else min)(r["baselines"].values(), key=lambda e: e[k])
                L.append(f"- {name}: significantly better than {len(good)}/12 baselines; significantly worse than "
                         f"{', '.join(bad) if bad else 'none'}; best baseline {best['id']} ({best['name']}) {best[k]:.3f}.")
            L.append("")
        L += ["Full per-baseline table: paper/generated/table_main.tex and table_main_tests.tex.", ""]

    rc = {d: j(ROOT / "results" / f"main_ragverify_{d}" / "risk_control.json") for d in DS}
    if all(rc.values()):
        L += ["## H4 — risk control (100 random 50/50 splits per pool)", "",
              "| Dataset | α | CRC realised risk (mean ± sd) | CRC coverage | CRC P(risk>α) | LTT error | LTT coverage | LTT P(error>α) |",
              "|---|---|---|---|---|---|---|---|"]
        for d, r in rc.items():
            for a in ("0.05", "0.1", "0.2"):
                c, t = r[a]["crc"], r[a]["ltt"]
                L.append(f"| {DS[d]} | {a} | {c['risk_mean']:.3f} ± {c['risk_sd']:.3f} | {c['coverage_mean']:.2f} | {c['frac_risk_gt_alpha']:.2f} | "
                         f"{t['risk_mean']:.3f} | {t['coverage_mean']:.2f} | {t['frac_risk_gt_alpha']:.2f} |")
        L += ["", "CRC guarantees expected risk ≤ α (so about half of the splits can exceed α); LTT guarantees P(error > α) ≤ δ = 0.1.", ""]

    p4 = j(A / "phase4_stats.json")
    if p4:
        L += ["## H2, H3 and ablations (first 500 items; variant minus main configuration)", ""]
        for d, o in p4["ablations"].items():
            h2, h3 = o.get("H2"), o.get("H3")
            if h2:
                L.append(f"- **H2 {DS[d]}** (external verifier minus self-verification): ΔAURC {ci(h2['aurc'])}, ΔH@0.8 {ci(h2['halluc@0.8'])}; "
                         f"LLM calls/example {h2['cost_main']['llm_calls']:.2f} vs {h2['cost_selfverify']['llm_calls']:.2f}.")
            if h3:
                g = o["variants"]["plusM1g09"]
                L.append(f"- **H3 {DS[d]}** (gate on minus gate off): ΔAcc {ci(g['diff_vs_main']['accuracy'])}; LLM calls {g['cost']['llm_calls']:.2f} vs "
                         f"{o['main']['cost']['llm_calls']:.2f}; retrievals {g['cost']['retrievals']:.2f} vs {o['main']['cost']['retrievals']:.2f} → "
                         f"H3 {'holds' if h3['holds'] else 'does not hold'}.")
        L += ["", "| Dataset | Variant | ΔAcc | ΔH@0.8 | ΔAURC |", "|---|---|---|---|---|"]
        for d, o in p4["ablations"].items():
            for v, e in o["variants"].items():
                dd = e["diff_vs_main"]
                L.append(f"| {DS[d]} | {v} | {ci(dd['accuracy'])} | {ci(dd['halluc@0.8'])} | {ci(dd['aurc'])} |")
        L += ["", "### Secondary backbone (Llama-3.2-3B Q4_K_M, 500 items)", ""]
        for d, o in p4["secondary"].items():
            rv = o["ragverify"]
            sig = [b for b, e in o["baselines"].items() if e["diff"]["halluc@0.8"]["p_holm"] < 0.05 and e["diff"]["halluc@0.8"]["diff"] < 0]
            L.append(f"- {DS[d]}: {METHOD_NAME} Acc {rv['accuracy']:.3f}, H@0.8 {rv['halluc@0.8']:.3f}, AURC {rv['aurc']:.3f}; H@0.8 significantly lower than "
                     f"{len(sig)}/{len(o['baselines'])} baselines ({', '.join(sig)}).")
        L += ["", "### Quantization check (Llama-3.2-3B Q8_0 minus Q4_K_M, first 200 items)", ""]
        for d, o in p4["quant"].items():
            for m, e in o.items():
                L.append(f"- {DS[d]} {m}: ΔAcc {ci(e['diff_q8_minus_q4']['accuracy'])}, ΔH@0.8 {ci(e['diff_q8_minus_q4']['halluc@0.8'])}")
        L.append("")

    ds_ = j(A / "distractor_stats.json")
    if ds_:
        L += ["## HotpotQA distractor robustness (300 items; Acc / H@0.8)", "", "| Method | 0% | 25% | 50% | 75% | Acc change 0→75% |",
              "|---|---|---|---|---|---|"]
        for m, r in ds_["methods"].items():
            L.append(f"| {m} | " + " | ".join(f"{r[p]['accuracy']:.3f} / {r[p]['halluc@0.8']:.3f}" for p in ("0", "25", "50", "75")) +
                     f" | {ci(r['75']['acc_change_vs_0'])} |")
        L.append("")

    rt = j(ROOT / "analysis" / "verifier_ragtruth_test.json")
    if rt:
        L += ["## Verifier component evaluation (RAGTruth QA test, 900 responses)", "", "| Verifier | AUROC | F1 at dev threshold |", "|---|---|---|"]
        L += [f"| {k} | {v['auroc']:.3f} | {v['f1_at_dev_threshold']:.3f} |" for k, v in rt.items()]
        L.append("")

    fa = j(A / "faithfulness.json")
    if fa:
        L += ["## Faithfulness (unsupported-answer rate among answered items; AlignScore-large / DeBERTa-v3-large NLI)", "",
              "| Method | " + " | ".join(DS[d] for d in ("popqa_lt", "triviaqa", "hotpotqa", "rgb")) + " |", "|---|---|---|---|---|"]
        ms_ = sorted({m for ev in fa.values() for d in ev.values() for m in d})
        for m in ms_:
            cells = []
            for d in ("popqa_lt", "triviaqa", "hotpotqa", "rgb"):
                a, b = fa.get("alignscore-large", {}).get(d, {}).get(m), fa.get("deberta-v3-l-nli", {}).get(d, {}).get(m)
                cells.append(f"{a['unsupported_rate']:.3f} / {b['unsupported_rate']:.3f}" if a and b else "--")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Deviations", "", "All deviations from the frozen protocol are dated entries in PROTOCOL.md §17 (also paper/generated/appendix_deviations.tex).", ""]
    (ROOT / "RESULTS_SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("RESULTS_SUMMARY.md", len(L), "lines")


if __name__ == "__main__":
    main()
