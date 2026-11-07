# RESULTS_SUMMARY (generated 2026-11-07 10:10 by analysis/make_results_summary.py, git e852746)

All numbers below are read from results/ by script. Primary backbone: Qwen2.5-7B-Instruct Q4_K_M. Acc = accuracy (abstention off); H@0.8 = hallucination rate (answered & wrong / all) at coverage 0.8; AURC = area under the risk-coverage curve (lower is better). Differences are RAG-Verify minus baseline with 95% paired-bootstrap CIs; p values are Holm-adjusted over the 12 baselines within dataset x metric (PROTOCOL §9).

## H1 — main test sets (primary backbone)

**PopQA-LT** (n = 1000): VerAbstain Acc 0.658, H@0.8 0.202, AURC 0.153. H1 **holds**.
- Acc: significantly better than 12/12 baselines; significantly worse than none; best baseline B3 (RAG + reranker) 0.618.
- H@0.8: significantly better than 12/12 baselines; significantly worse than none; best baseline B3 (RAG + reranker) 0.259.
- AURC: significantly better than 12/12 baselines; significantly worse than none; best baseline B3 (RAG + reranker) 0.216.

**TriviaQA** (n = 1000): VerAbstain Acc 0.676, H@0.8 0.173, AURC 0.139. H1 **does not hold**.
- Acc: significantly better than 5/12 baselines; significantly worse than none; best baseline B4 (RAG large-k) 0.699.
- H@0.8: significantly better than 11/12 baselines; significantly worse than none; best baseline B4 (RAG large-k) 0.186.
- AURC: significantly better than 11/12 baselines; significantly worse than none; best baseline B4 (RAG large-k) 0.153.

**HotpotQA** (n = 1000): VerAbstain Acc 0.395, H@0.8 0.430, AURC 0.469. H1 **does not hold**.
- Acc: significantly better than 10/12 baselines; significantly worse than none; best baseline B3 (RAG + reranker) 0.371.
- H@0.8: significantly better than 10/12 baselines; significantly worse than none; best baseline B3 (RAG + reranker) 0.460.
- AURC: significantly better than 6/12 baselines; significantly worse than none; best baseline B3 (RAG + reranker) 0.476.

**PubHealth** (n = 987): VerAbstain Acc 0.532, H@0.8 0.362, AURC 0.407. H1 **does not hold**.
- Acc: significantly better than 8/12 baselines; significantly worse than B8; best baseline B8 (Self-RAG) 0.716.
- H@0.8: significantly better than 6/12 baselines; significantly worse than B8; best baseline B8 (Self-RAG) 0.234.
- AURC: significantly better than 7/12 baselines; significantly worse than B8; best baseline B8 (Self-RAG) 0.317.

**RGB** (n = 1300): VerAbstain Acc 0.625, H@0.8 0.264, AURC 0.341. H1 **does not hold**.
- Acc: significantly better than 4/12 baselines; significantly worse than B6; best baseline B6 (Chain-of-Note) 0.694.
- H@0.8: significantly better than 3/12 baselines; significantly worse than B6; best baseline B6 (Chain-of-Note) 0.227.
- AURC: significantly better than 2/12 baselines; significantly worse than B2, B3, B4, B6, B7, B8, B9, A1; best baseline B6 (Chain-of-Note) 0.255.

Full per-baseline table: paper/generated/table_main.tex and table_main_tests.tex.

## H4 — risk control (100 random 50/50 splits per pool)

| Dataset | α | CRC realised risk (mean ± sd) | CRC coverage | CRC P(risk>α) | LTT error | LTT coverage | LTT P(error>α) |
|---|---|---|---|---|---|---|---|
| PopQA-LT | 0.05 | 0.052 ± 0.013 | 0.44 | 0.55 | 0.004 | 0.01 | 0.07 |
| PopQA-LT | 0.1 | 0.101 ± 0.018 | 0.61 | 0.51 | 0.024 | 0.04 | 0.03 |
| PopQA-LT | 0.2 | 0.200 ± 0.023 | 0.79 | 0.48 | 0.096 | 0.34 | 0.08 |
| TriviaQA | 0.05 | 0.051 ± 0.014 | 0.45 | 0.49 | 0.002 | 0.00 | 0.02 |
| TriviaQA | 0.1 | 0.099 ± 0.017 | 0.64 | 0.44 | 0.028 | 0.05 | 0.04 |
| TriviaQA | 0.2 | 0.200 ± 0.024 | 0.83 | 0.53 | 0.156 | 0.62 | 0.12 |
| HotpotQA | 0.05 | 0.051 ± 0.015 | 0.13 | 0.40 | 0.000 | 0.00 | 0.00 |
| HotpotQA | 0.1 | 0.100 ± 0.020 | 0.25 | 0.41 | 0.000 | 0.00 | 0.00 |
| HotpotQA | 0.2 | 0.198 ± 0.023 | 0.43 | 0.37 | 0.008 | 0.00 | 0.02 |
| PubHealth | 0.05 | 0.048 ± 0.015 | 0.14 | 0.36 | 0.000 | 0.00 | 0.00 |
| PubHealth | 0.1 | 0.102 ± 0.021 | 0.26 | 0.48 | 0.000 | 0.00 | 0.00 |
| PubHealth | 0.2 | 0.203 ± 0.027 | 0.48 | 0.47 | 0.000 | 0.00 | 0.00 |
| RGB | 0.05 | 0.049 ± 0.012 | 0.12 | 0.41 | 0.000 | 0.00 | 0.00 |
| RGB | 0.1 | 0.100 ± 0.017 | 0.32 | 0.42 | 0.000 | 0.00 | 0.00 |
| RGB | 0.2 | 0.198 ± 0.019 | 0.65 | 0.41 | 0.000 | 0.00 | 0.00 |

CRC guarantees expected risk ≤ α (so about half of the splits can exceed α); LTT guarantees P(error > α) ≤ δ = 0.1.

## H2, H3 and ablations (first 500 items; variant minus main configuration)

- **H2 PopQA-LT** (external verifier minus self-verification): ΔAURC +0.000 [-0.013, +0.013], ΔH@0.8 +0.016 [-0.008, +0.040]; LLM calls/example 2.38 vs 4.13.
- **H3 PopQA-LT** (gate on minus gate off): ΔAcc -0.082 [-0.108, -0.058]; LLM calls 2.07 vs 2.38; retrievals 0.83 vs 1.12 → H3 does not hold.
- **H2 HotpotQA** (external verifier minus self-verification): ΔAURC +0.017 [-0.015, +0.049], ΔH@0.8 +0.006 [-0.028, +0.040]; LLM calls/example 2.88 vs 5.33.
- **H3 HotpotQA** (gate on minus gate off): ΔAcc -0.062 [-0.092, -0.034]; LLM calls 2.25 vs 2.88; retrievals 0.87 vs 1.34 → H3 does not hold.

| Dataset | Variant | ΔAcc | ΔH@0.8 | ΔAURC |
|---|---|---|---|---|
| PopQA-LT | plusM1g09 | -0.082 [-0.108, -0.058] | +0.080 [+0.054, +0.108] | +0.110 [+0.072, +0.150] |
| PopQA-LT | noM2 | -0.102 [-0.138, -0.066] | +0.076 [+0.040, +0.114] | +0.068 [+0.037, +0.101] |
| PopQA-LT | noM3 | -0.046 [-0.070, -0.024] | +0.046 [+0.018, +0.076] | +0.040 [+0.020, +0.063] |
| PopQA-LT | noM4 | -0.032 [-0.050, -0.016] | +0.000 [-0.018, +0.018] | +0.016 [+0.007, +0.026] |
| PopQA-LT | noM5 | +0.000 [-0.008, +0.008] | +0.002 [-0.008, +0.012] | +0.003 [-0.001, +0.007] |
| PopQA-LT | selfverify | +0.004 [-0.010, +0.018] | -0.016 [-0.040, +0.008] | -0.000 [-0.013, +0.013] |
| PopQA-LT | hhem | +0.000 [-0.008, +0.008] | -0.008 [-0.022, +0.004] | +0.008 [-0.001, +0.017] |
| PopQA-LT | minicheckdeb | +0.000 [-0.008, +0.008] | -0.004 [-0.020, +0.012] | +0.010 [+0.002, +0.019] |
| PopQA-LT | k3 | -0.008 [-0.030, +0.016] | +0.002 [-0.026, +0.030] | +0.006 [-0.011, +0.023] |
| PopQA-LT | k10 | -0.002 [-0.026, +0.020] | -0.002 [-0.028, +0.026] | -0.008 [-0.023, +0.007] |
| PopQA-LT | rounds1 | +0.000 [-0.006, +0.006] | +0.002 [+0.000, +0.006] | +0.001 [-0.000, +0.003] |
| HotpotQA | plusM1g09 | -0.062 [-0.092, -0.034] | +0.066 [+0.034, +0.100] | +0.119 [+0.070, +0.176] |
| HotpotQA | noM2 | -0.030 [-0.062, +0.002] | +0.022 [-0.018, +0.062] | +0.029 [-0.010, +0.069] |
| HotpotQA | noM3 | -0.020 [-0.054, +0.012] | +0.022 [-0.020, +0.064] | +0.033 [-0.006, +0.073] |
| HotpotQA | noM4 | -0.038 [-0.058, -0.020] | +0.026 [-0.004, +0.056] | +0.008 [-0.024, +0.038] |
| HotpotQA | noM5 | -0.010 [-0.024, +0.002] | +0.008 [-0.014, +0.030] | -0.006 [-0.029, +0.016] |
| HotpotQA | selfverify | +0.008 [-0.012, +0.028] | -0.006 [-0.040, +0.028] | -0.017 [-0.049, +0.015] |
| HotpotQA | hhem | -0.006 [-0.018, +0.006] | +0.008 [-0.012, +0.028] | +0.001 [-0.019, +0.022] |
| HotpotQA | minicheckdeb | -0.006 [-0.018, +0.006] | +0.010 [-0.014, +0.034] | +0.008 [-0.017, +0.033] |
| HotpotQA | k3 | -0.026 [-0.052, +0.000] | +0.016 [-0.018, +0.050] | +0.014 [-0.019, +0.048] |
| HotpotQA | k10 | -0.012 [-0.038, +0.014] | +0.014 [-0.020, +0.048] | +0.007 [-0.026, +0.038] |
| HotpotQA | rounds1 | +0.004 [-0.004, +0.012] | -0.004 [-0.016, +0.006] | -0.013 [-0.025, -0.003] |

### Secondary backbone (Llama-3.2-3B Q4_K_M, 500 items)

- PopQA-LT: VerAbstain Acc 0.570, H@0.8 0.262, AURC 0.220; H@0.8 significantly lower than 8/8 baselines (B1, B2, B3, B5, B7, B9, A1, A2).
- HotpotQA: VerAbstain Acc 0.310, H@0.8 0.508, AURC 0.536; H@0.8 significantly lower than 2/8 baselines (B1, B9).
- PubHealth: VerAbstain Acc 0.528, H@0.8 0.344, AURC 0.305; H@0.8 significantly lower than 7/8 baselines (B1, B2, B5, B7, B9, A1, A2).

### Quantization check (Llama-3.2-3B Q8_0 minus Q4_K_M, first 200 items)

- PopQA-LT closed_book: ΔAcc +0.020 [+0.005, +0.040], ΔH@0.8 -0.020 [-0.075, +0.035]
- PopQA-LT rag_topk: ΔAcc +0.015 [-0.010, +0.040], ΔH@0.8 +0.005 [-0.045, +0.055]
- PopQA-LT ragverify: ΔAcc +0.015 [-0.020, +0.050], ΔH@0.8 +0.015 [-0.030, +0.060]
- HotpotQA closed_book: ΔAcc -0.005 [-0.035, +0.025], ΔH@0.8 +0.010 [-0.050, +0.070]
- HotpotQA rag_topk: ΔAcc +0.005 [-0.020, +0.030], ΔH@0.8 -0.005 [-0.055, +0.045]
- HotpotQA ragverify: ΔAcc +0.005 [-0.035, +0.045], ΔH@0.8 -0.015 [-0.065, +0.035]

## HotpotQA distractor robustness (300 items; Acc / H@0.8)

| Method | 0% | 25% | 50% | 75% | Acc change 0→75% |
|---|---|---|---|---|---|
| ragverify | 0.347 / 0.480 | 0.373 / 0.447 | 0.300 / 0.510 | 0.240 / 0.573 | -0.107 [-0.157, -0.060] |
| rag_topk | 0.333 / 0.500 | 0.357 / 0.473 | 0.290 / 0.543 | 0.233 / 0.573 | -0.100 [-0.143, -0.057] |
| rag_rerank | 0.333 / 0.487 | 0.357 / 0.480 | 0.280 / 0.540 | 0.240 / 0.583 | -0.093 [-0.140, -0.050] |
| crag | 0.333 / 0.500 | 0.317 / 0.510 | 0.297 / 0.527 | 0.233 / 0.580 | -0.100 [-0.147, -0.053] |
| chain_of_note | 0.310 / 0.523 | 0.313 / 0.540 | 0.250 / 0.577 | 0.180 / 0.643 | -0.130 [-0.180, -0.080] |
| selfrag | 0.347 / 0.517 | 0.353 / 0.503 | 0.317 / 0.537 | 0.280 / 0.567 | -0.067 [-0.107, -0.027] |

## Verifier component evaluation (RAGTruth QA test, 900 responses)

| Verifier | AUROC | F1 at dev threshold |
|---|---|---|
| minicheck-ft5 | 0.848 | 0.569 |
| minicheck-deb | 0.826 | 0.520 |
| hhem-2.1-open | 0.874 | 0.627 |
| deberta-v3-l-nli | 0.731 | 0.401 |

## Faithfulness (unsupported-answer rate among answered items; AlignScore-large / DeBERTa-v3-large NLI)

| Method | PopQA-LT | TriviaQA | HotpotQA | RGB |
|---|---|---|---|---|
| chain_of_note | 0.091 / 0.141 | 0.125 / 0.113 | 0.340 / 0.064 | 0.123 / 0.273 |
| cove | 0.354 / 0.357 | 0.270 / 0.263 | 0.481 / 0.183 | 0.271 / 0.396 |
| crag | 0.168 / 0.271 | 0.216 / 0.273 | 0.451 / 0.157 | 0.091 / 0.355 |
| flare | 0.189 / 0.170 | 0.212 / 0.181 | 0.421 / 0.102 | 0.132 / 0.244 |
| rag_largek | 0.244 / 0.267 | 0.217 / 0.226 | 0.449 / 0.166 | 0.158 / 0.296 |
| rag_rerank | 0.161 / 0.148 | 0.201 / 0.132 | 0.421 / 0.114 | 0.146 / 0.142 |
| rag_topk | 0.215 / 0.238 | 0.202 / 0.198 | 0.442 / 0.131 | 0.158 / 0.296 |
| ragverify | 0.116 / 0.120 | 0.163 / 0.097 | 0.353 / 0.091 | 0.118 / 0.135 |
| rarr | 0.173 / 0.277 | 0.223 / 0.289 | 0.427 / 0.186 | 0.171 / 0.416 |
| selfrag | 0.160 / 0.083 | 0.449 / 0.183 | 0.552 / 0.228 | 0.285 / 0.196 |

## Deviations

All deviations from the frozen protocol are dated entries in PROTOCOL.md §17 (also paper/generated/appendix_deviations.tex).
