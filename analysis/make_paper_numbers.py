"""Generate every number used in the prose of the paper (Rule 2), plus the remaining generated tables and figures.

Outputs:
  paper/generated/results_macros.tex   \\R{key} lookup; an unknown key prints a bold ??key in the PDF and a LaTeX warning
  paper/generated/table_risk_control.tex, table_ragtruth.tex, table_faithfulness_ci.tex, table_distractor_pairs.tex
  paper/figures/risk_coverage.pdf
  results/analysis/paper_numbers.json  (the same key/value pairs, for auditing)
Inputs: results/analysis/{main_stats,phase4_stats,distractor_stats,faithfulness}.json, results/main_ragverify_*/risk_control.json,
analysis/verifier_ragtruth_test.json, results/main_*/summary.json and scores.jsonl, results/rob_*/scores.jsonl.
Faithfulness CIs are Wilson 95% intervals per cell (unpaired: methods answer different subsets; descriptive only).
Usage: .venv/Scripts/python.exe analysis/make_paper_numbers.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "src"))
from main_stats import BASELINES, DATASETS, DS_NAME, load, load_run  # noqa: E402
from ragverify.metrics import mcnemar_exact, paired_bootstrap  # noqa: E402
from method_name import METHOD_NAME  # noqa: E402

A = ROOT / "results" / "analysis"
G = ROOT / "paper" / "generated"
SHORT = {"popqa_lt": "pq", "triviaqa": "tq", "hotpotqa": "hq", "pubhealth": "ph", "rgb": "rgb"}
MET = {"accuracy": "acc", "halluc@0.8": "hE", "aurc": "aurc"}  # hE = hallucination at coverage 0.8
FA_NAME = {"ragverify": METHOD_NAME}
for _b, _n, _m in BASELINES:  # first id wins (A1 reuses B2's run rag_topk)
    FA_NAME.setdefault(_m, f"{_b} {_n}")
R: dict[str, str] = {}


def j(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def f3(x: float) -> str:
    return f"{x:.3f}"


def s3(x: float) -> str:
    return f"{x:+.3f}".replace("-", "$-$")


def ci(e: dict, scale: float = 1.0) -> str:
    return f"{s3(e['diff'] * scale)} [{s3(e['ci_low'] * scale)}, {s3(e['ci_high'] * scale)}]"


def pfmt(p: float, n_resample: int | None = None) -> str:
    if p == 0 and n_resample:
        return f"$p<10^{{-{int(math.log10(n_resample))}}}$"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"$p={m}\\times10^{{{int(e)}}}$"
    return f"$p={p:.3f}$"


def wilson(k: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main_numbers() -> None:
    ms = j(A / "main_stats.json")["datasets"]
    holds = 0
    for d, r in ms.items():
        s = SHORT[d]
        rv = r["ragverify"]
        R[f"{s}-n"] = f"{rv['n']:,}"
        for k, mk in MET.items():
            R[f"{s}-rv-{mk}"] = f3(rv[k])
        holds += r["H1_holds"]
        R[f"{s}-h1"] = "holds" if r["H1_holds"] else "does not hold"
        for b, e in r["baselines"].items():
            for k, mk in MET.items():
                R[f"{s}-{b}-{mk}"] = f3(e[k])
                dd = e["diff"][k]
                R[f"{s}-{b}-d{mk}"] = ci(dd)
                R[f"{s}-{b}-d{mk}-p"] = pfmt(dd["p_holm"], 10_000 if k != "accuracy" else None)
        for k, mk in MET.items():
            up = k == "accuracy"
            good = [b for b, e in r["baselines"].items() if e["diff"][k]["p_holm"] < 0.05 and (e["diff"][k]["diff"] > 0) == up]
            bad = [b for b, e in r["baselines"].items() if e["diff"][k]["p_holm"] < 0.05 and (e["diff"][k]["diff"] > 0) != up]
            ns = [b for b in r["baselines"] if b not in good and b not in bad]
            R[f"{s}-{mk}-nbetter"] = str(len(good))
            R[f"{s}-{mk}-nworse"] = str(len(bad))
            R[f"{s}-{mk}-worse"] = ", ".join(bad) if bad else "none"
            R[f"{s}-{mk}-ns"] = ", ".join(ns) if ns else "none"
            best = (max if up else min)(r["baselines"].values(), key=lambda e: e[k])
            R[f"{s}-{mk}-best"] = f"{best['id']} ({best['name']})"
            R[f"{s}-{mk}-bestid"] = best["id"]
            R[f"{s}-{mk}-bestval"] = f3(best[k])
            R[f"{s}-{mk}-rank"] = str(1 + sum((e[k] > rv[k]) if up else (e[k] < rv[k]) for e in r["baselines"].values()))
    R["h1-nholds"] = str(holds)
    words = ["no", "one", "two", "three", "four", "five"]
    for mk in MET.values():
        n1 = sum(R[f"{SHORT[d]}-{mk}-rank"] == "1" for d in ms)
        R[f"{mk}-nbest"], R[f"{mk}-nbest-w"] = str(n1), words[n1]
    R["h1-nholds-w"] = words[holds]
    pool = [j(ROOT / "results" / f"main_ragverify_{d}" / "risk_control.json")["n"] for d in DATASETS]
    R["rc-cal-min"], R["rc-cal-max"] = str(min(pool) // 2), str(max(pool) // 2)
    # A2 seed variability (A2 uses sampled answers; seed 0 is used for tests)
    # efficiency (mean over the five datasets, each weighted equally), RAG-Verify vs B2 and the most expensive baselines
    for m, key in (("ragverify", "rv"), ("rag_topk", "B2"), ("selfrag", "B8"), ("cove", "B9"), ("rarr", "B10"), ("chain_of_note", "B6"),
                   ("crag", "B5"), ("rag_selfcheck", "A2"), ("closed_book", "B1")):
        sm = [j(ROOT / "results" / f"main_{m}_{d}" / "summary.json") for d in DATASETS]
        for k, nm, p in (("mean_llm_calls", "calls", 2), ("mean_retrievals", "retr", 2), ("mean_llm_seconds", "sec", 1),
                         ("mean_prompt_tokens", "ptok", 0), ("mean_completion_tokens", "ctok", 0)):
            R[f"eff-{key}-{nm}"] = f"{sum(x[k] for x in sm) / len(sm):.{p}f}"


def risk_control() -> None:
    L = ["% generated by analysis/make_paper_numbers.py from results/main_ragverify_*/risk_control.json",
         "\\begin{tabular}{@{}llcccccc@{}}", "\\toprule & & \\multicolumn{3}{c}{CRC (expected risk $\\le\\alpha$)} & "
         "\\multicolumn{3}{c}{LTT ($P(\\text{risk}>\\alpha)\\le\\delta=0.1$)} \\\\ \\cmidrule(lr){3-5}\\cmidrule(l){6-8}",
         "Dataset & $\\alpha$ & Risk (mean $\\pm$ sd) & Coverage & Frac.\\ $>\\alpha$ & Risk & Coverage & Frac.\\ $>\\alpha$ \\\\ \\midrule"]
    for d in DATASETS:
        r = j(ROOT / "results" / f"main_ragverify_{d}" / "risk_control.json")
        R[f"{SHORT[d]}-rc-npool"] = f"{r['n']:,}"
        for a in ("0.05", "0.1", "0.2"):
            c, t = r[a]["crc"], r[a]["ltt"]
            ak = {"0.05": "05", "0.1": "10", "0.2": "20"}[a]
            p = f"{SHORT[d]}-rc{ak}"
            R[f"{p}-crc-risk"], R[f"{p}-crc-sd"] = f3(c["risk_mean"]), f3(c["risk_sd"])
            R[f"{p}-crc-cov"], R[f"{p}-crc-frac"] = f"{c['coverage_mean']:.2f}", f"{c['frac_risk_gt_alpha']:.2f}"
            R[f"{p}-ltt-risk"], R[f"{p}-ltt-cov"], R[f"{p}-ltt-frac"] = f3(t["risk_mean"]), f"{t['coverage_mean']:.2f}", f"{t['frac_risk_gt_alpha']:.2f}"
            L.append(f"{DS_NAME[d] if a == '0.05' else ''} & {a} & {f3(c['risk_mean'])} $\\pm$ {f3(c['risk_sd'])} & {c['coverage_mean']:.2f} & "
                     f"{c['frac_risk_gt_alpha']:.2f} & {f3(t['risk_mean'])} & {t['coverage_mean']:.2f} & {t['frac_risk_gt_alpha']:.2f} \\\\")
        if d != DATASETS[-1]:
            L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}"]
    (G / "table_risk_control.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def phase4() -> None:
    p4 = j(A / "phase4_stats.json")
    for d, o in p4["ablations"].items():
        s = SHORT[d]
        R[f"abl-{s}-n"] = str(o["main"]["n"])
        for k in ("llm_calls", "retrievals"):
            R[f"abl-{s}-main-{'calls' if k == 'llm_calls' else 'retr'}"] = f"{o['main']['cost'][k]:.2f}"
        for v, e in o["variants"].items():
            for k, mk in MET.items():
                R[f"abl-{s}-{v}-d{mk}"] = ci(e["diff_vs_main"][k])
            for k in ("llm_calls", "retrievals"):
                R[f"abl-{s}-{v}-{'calls' if k == 'llm_calls' else 'retr'}"] = f"{e['cost'][k]:.2f}"
        h2 = o["H2"]
        R[f"h2-{s}-daurc"], R[f"h2-{s}-dhE"] = ci(h2["aurc"]), ci(h2["halluc@0.8"])
        R[f"h2-{s}-daurc-p"], R[f"h2-{s}-dhE-p"] = pfmt(h2["aurc"]["p"]), pfmt(h2["halluc@0.8"]["p"])
        cm, cs = h2["cost_main"], h2["cost_selfverify"]
        R[f"h2-{s}-calls-main"], R[f"h2-{s}-calls-self"] = f"{cm['llm_calls']:.2f}", f"{cs['llm_calls']:.2f}"
        R[f"h2-{s}-calls-saving"] = f"{100 * (1 - cm['llm_calls'] / cs['llm_calls']):.0f}"
        R[f"h2-{s}-sec-main"], R[f"h2-{s}-sec-self"] = f"{cm['llm_seconds']:.1f}", f"{cs['llm_seconds']:.1f}"
        R[f"h3-{s}"] = "holds" if o["H3"]["holds"] else "does not hold"
    for d, o in p4["secondary"].items():
        s = SHORT[d]
        rv = o["ragverify"]
        for k, mk in MET.items():
            R[f"sec-{s}-rv-{mk}"] = f3(rv[k])
        R[f"sec-{s}-n"] = str(rv["n"])
        R[f"sec-{s}-nbase"] = str(len(o["baselines"]))
        sig = [b for b, e in o["baselines"].items() if e["diff"]["halluc@0.8"]["p_holm"] < 0.05 and e["diff"]["halluc@0.8"]["diff"] < 0]
        R[f"sec-{s}-hE-nbetter"] = str(len(sig))
        R[f"sec-{s}-hE-ns"] = ", ".join(b for b in o["baselines"] if b not in sig) or "none"
        R[f"sec-{s}-hE-lowest"] = "yes" if all(rv["halluc@0.8"] <= e["halluc@0.8"] for e in o["baselines"].values()) else "no"
        bestacc = max(o["baselines"].items(), key=lambda kv: kv[1]["accuracy"])
        R[f"sec-{s}-acc-best"], R[f"sec-{s}-acc-bestval"] = bestacc[0], f3(bestacc[1]["accuracy"])
        sigacc = [b for b, e in o["baselines"].items() if e["diff"]["accuracy"]["p_holm"] < 0.05 and e["diff"]["accuracy"]["diff"] > 0]
        R[f"sec-{s}-acc-nbetter"] = str(len(sigacc))
    nci = 0
    tot = 0
    for d, o in p4["quant"].items():
        for m, e in o.items():
            for k, mk in MET.items():
                dd = e["diff_q8_minus_q4"][k]
                R[f"q-{SHORT[d]}-{m}-d{mk}"] = ci(dd)
                if k != "aurc":
                    tot += 1
                    nci += (dd["ci_low"] > 0) or (dd["ci_high"] < 0)
    R["q-nexcl"], R["q-ntot"] = str(nci), str(tot)


def distractor() -> None:
    ds = j(A / "distractor_stats.json")["methods"]
    for m, r in ds.items():
        for p in ("0", "25", "50", "75"):
            R[f"dist-{m}-{p}-acc"], R[f"dist-{m}-{p}-hE"] = f3(r[p]["accuracy"]), f3(r[p]["halluc@0.8"])
        R[f"dist-{m}-drop"] = ci(r["75"]["acc_change_vs_0"])
    R["dist-n"] = str(ds["ragverify"]["0"]["n"])
    # RAG-Verify minus each method, accuracy, at each distractor level (paired bootstrap + McNemar, Holm over methods per level)
    L = ["% generated by analysis/make_paper_numbers.py: RAG-Verify minus method, accuracy, paired bootstrap 95% CI, McNemar exact, "
         "Holm over the 5 methods within each distractor level (exploratory; not a pre-registered hypothesis test)",
         "\\begin{tabular}{@{}lcccc@{}}", "\\toprule vs. & 0\\% & 25\\% & 50\\% & 75\\% \\\\ \\midrule"]
    others = [m for m in ds if m != "ragverify"]
    cells = {m: [] for m in others}
    for p in (0, 25, 50, 75):
        _, ids, rv, _ = load_run(f"rob_ragverify_hotpotqa_dist{p:02d}", "ragverify")
        res = {}
        for m in others:
            _, ids2, c2, _ = load_run(f"rob_{m}_hotpotqa_dist{p:02d}", m)
            assert ids == ids2
            res[m] = (paired_bootstrap(rv, c2), mcnemar_exact(rv, c2)["p"])
        order = sorted(others, key=lambda m: res[m][1])
        adj, run = {}, 0.0
        for i, m in enumerate(order):
            run = max(run, min(1.0, (len(order) - i) * res[m][1]))
            adj[m] = run
        for m in others:
            b = res[m][0]
            e = {"diff": b["diff"], "ci_low": b["ci_low"], "ci_high": b["ci_high"]}
            R[f"distpair-{m}-{p}"] = ci(e)
            R[f"distpair-{m}-{p}-p"] = pfmt(adj[m])
            cells[m].append(ci(e) + ("$^{*}$" if adj[m] < 0.05 else ""))
    for m in others:
        L.append(f"{FA_NAME.get(m, m)} & " + " & ".join(cells[m]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (G / "table_distractor_pairs.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def ragtruth() -> None:
    rt = j(ROOT / "analysis" / "verifier_ragtruth_test.json")
    nice = {"minicheck-ft5": "MiniCheck-Flan-T5-L (M4, used)", "minicheck-deb": "MiniCheck-DeBERTa-L", "hhem-2.1-open": "HHEM-2.1-Open",
            "deberta-v3-l-nli": "DeBERTa-v3-L NLI"}
    L = ["% generated by analysis/make_paper_numbers.py from analysis/verifier_ragtruth_test.json",
         "\\begin{tabular}{@{}lcc@{}}", "\\toprule Verifier & AUROC & F1 at dev threshold \\\\ \\midrule"]
    for k, v in rt.items():
        L.append(f"{nice.get(k, k)} & {f3(v['auroc'])} & {f3(v['f1_at_dev_threshold'])} \\\\")
        R[f"rt-{k}-auroc"], R[f"rt-{k}-f1"] = f3(v["auroc"]), f3(v["f1_at_dev_threshold"])
        R["rt-n"], R["rt-pos"] = f"{v['n']:,}", f"{v['positive_rate']:.2f}"
    L += ["\\bottomrule", "\\end{tabular}"]
    (G / "table_ragtruth.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def faithfulness() -> None:
    fa = j(A / "faithfulness.json")
    dss = ["popqa_lt", "triviaqa", "hotpotqa", "rgb"]
    L = ["% generated by analysis/make_paper_numbers.py: unsupported-answer rate among answered items with Wilson 95% CI "
         "(unpaired; descriptive) and the number of answered items",
         "\\begin{tabular}{@{}llcccc@{}}", "\\toprule Judge & Method & " + " & ".join(DS_NAME[d] for d in dss) + " \\\\ \\midrule"]
    for jn, judge in (("AlignScore-L", "alignscore-large"), ("DeBERTa NLI", "deberta-v3-l-nli")):
        jk = "as" if judge.startswith("align") else "nli"
        for i, m in enumerate(fa[judge]["popqa_lt"]):
            cells = []
            for d in dss:
                e = fa[judge][d][m]
                lo, hi = wilson(round(e["unsupported_rate"] * e["n_answered"]), e["n_answered"])
                cells.append(f"{e['unsupported_rate']:.3f} [{lo:.3f}, {hi:.3f}] ({e['n_answered']})")
                R[f"fa-{jk}-{SHORT[d]}-{m}"] = f3(e["unsupported_rate"])
                R[f"fa-{SHORT[d]}-{m}-nans"] = str(e["n_answered"])
            L.append(f"{jn if i == 0 else ''} & {FA_NAME[m]} & " + " & ".join(cells) + " \\\\")
        if jk == "as":
            L.append("\\addlinespace")
        for d in dss:
            rates = {m: e["unsupported_rate"] for m, e in fa[judge][d].items()}
            R[f"fa-{jk}-{SHORT[d]}-rank"] = str(1 + sum(v < rates["ragverify"] for v in rates.values()))
            best = min(rates, key=rates.get)
            R[f"fa-{jk}-{SHORT[d]}-best"] = FA_NAME[best]
    L += ["\\bottomrule", "\\end{tabular}"]
    (G / "table_faithfulness_ci.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def secondary_metrics() -> None:
    """PROTOCOL §8 secondary metrics: EM / F1 / native abstention (QA) and RGB per-condition rates (abstention off)."""
    meths = [(METHOD_NAME, "ragverify")] + [(f"{b} {n}", m) for b, n, m in BASELINES if b != "A1"]
    qa = ["popqa_lt", "triviaqa", "hotpotqa"]
    L = ["% generated by analysis/make_paper_numbers.py from results/main_*/scores.jsonl and summary.json: EM / F1 / native "
         "abstention rate (QA); RGB accuracy per noise rate, rejection rate on noise-1.0 items, accuracy and fake-answer rate on "
         "counterfactual items. A2 = seed 0.",
         "\\begin{tabular}{@{}l" + "c" * (len(qa) + 6) + "@{}}",
         "\\toprule & " + " & ".join(f"{DS_NAME[d]}" for d in qa) + " & \\multicolumn{6}{c}{RGB} \\\\ \\cmidrule(lr){2-4}\\cmidrule(l){5-10}",
         "Method & \\multicolumn{3}{c}{EM / F1 / abstain} & noise 0 & 0.4 & 0.8 & reject.\\,$\\uparrow$ & CF acc. & fake$\\downarrow$ \\\\ \\midrule"]
    for name, m in meths:
        cells = []
        for d in qa:
            s = j(ROOT / "results" / f"main_{m}_{d}" / "summary.json")
            cells.append(f"{s['em']:.2f} / {s['f1']:.2f} / {s['native_abstain_rate']:.2f}")
            R[f"sm-{SHORT[d]}-{m}-abst"] = f"{s['native_abstain_rate']:.3f}"
        rows = [json.loads(x) for x in (ROOT / "results" / f"main_{m}_rgb" / "scores.jsonl").open(encoding="utf-8")]
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r["id"].split(":")[1], []).append(r)
        vals = {c: float(np.mean([r["correct"] for r in by[c]])) for c in ("n0.0", "n0.4", "n0.8", "n1.0", "fact")}
        fake = float(np.mean([r.get("fake_answer", 0) for r in by["fact"]]))
        for c, v in vals.items():
            R[f"rgb-{m}-{c.replace('.', '')}"] = f3(v)
        R[f"rgb-{m}-fake"] = f3(fake)
        cells += [f"{vals['n0.0']:.2f}", f"{vals['n0.4']:.2f}", f"{vals['n0.8']:.2f}", f"{vals['n1.0']:.2f}", f"{vals['fact']:.2f}", f"{fake:.2f}"]
        L.append(f"{name} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (G / "table_secondary_metrics.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def captioned_tests_table() -> None:
    """Copy of analysis/main_stats.py's longtable with a caption and label inserted (longtable captions must be inside)."""
    lines = (G / "table_main_tests.tex").read_text(encoding="utf-8").splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("\\begin{longtable}"))
    cap = ("\\caption{\\method{} minus each baseline on the test sets: difference with 95\\% paired-bootstrap CI and "
           "Holm-adjusted $p$ value (McNemar exact test for accuracy, paired permutation test for H@0.8, bootstrap for AURC; "
           "0 means below the resolution of 10{,}000 resamples).}\\label{tab:main-tests}\\\\")
    lines.insert(i + 1, cap)
    (G / "table_main_tests_captioned.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def risk_coverage_fig() -> None:
    ms = j(A / "main_stats.json")["datasets"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharey=False)
    for ax, d in zip(axes, DATASETS):
        best = min(ms[d]["baselines"].values(), key=lambda e: e["aurc"])
        bm = next(m for b, n, m in BASELINES if b == best["id"])
        curves = [(METHOD_NAME, "ragverify", "-", "C3"), ("B2 RAG top-k (A1 score)", "rag_topk", "--", "C0")]
        if bm != "rag_topk":
            curves.append((f"{best['id']} (lowest baseline AURC)", bm, ":", "C1"))
        for lab, m, ls, col in curves:
            _, _, corr, sc = load(m, d)
            order = np.lexsort((np.arange(len(sc)), -sc))
            err = 1 - corr[order]
            k = np.arange(1, len(err) + 1)
            ax.plot(k / len(err), np.cumsum(err) / k, ls, color=col, label=lab, lw=1.6)
        ax.set_title(DS_NAME[d], fontsize=10)
        ax.set_xlabel("Coverage")
        ax.axvline(0.8, color="0.7", lw=0.8)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Risk (error rate among answered)")
    fig.tight_layout()
    fig.savefig(ROOT / "paper" / "figures" / "risk_coverage.pdf")


def write() -> None:
    def esc(v: str) -> str:
        return v.replace("%", "\\%").replace("&", "\\&")
    L = ["% generated by analysis/make_paper_numbers.py -- do not edit. Usage in the text: \\R{key}",
         "\\makeatletter",
         "\\newcommand{\\R}[1]{\\@ifundefined{rv@#1}{\\textbf{??#1}\\PackageWarning{results}{Undefined result key #1}}{\\@nameuse{rv@#1}}}"]
    L += [f"\\@namedef{{rv@{k}}}{{{esc(v)}}}" for k, v in sorted(R.items())]
    L.append("\\makeatother")
    (G / "results_macros.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    (A / "paper_numbers.json").write_text(json.dumps(R, indent=0, sort_keys=True), encoding="utf-8")
    print(len(R), "keys")


if __name__ == "__main__":
    main_numbers()
    risk_control()
    phase4()
    distractor()
    ragtruth()
    faithfulness()
    secondary_metrics()
    captioned_tests_table()
    risk_coverage_fig()
    write()
