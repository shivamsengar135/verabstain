# Gate 1 — summary (2026-11-05 02:00 IST, T0 + 4.2 h)

**Needed from you:** (1) GO / no-GO, and which of options A–B below; (2) is `github.com/shivamsengar135/RAG-Verify` (ICTCS 2026 submission) your work? (3) approve the RARR adaptation. Nothing has touched a test set yet. `PROTOCOL.md` is complete but **not frozen**.

**Method (RAG-Verify, as tuned on dev).** Closed-book draft with a token-probability gate (answer directly if confidence ≥ 0.9) → hybrid BM25 + bge-small retrieval, reciprocal rank fusion, MiniLM cross-encoder rerank → grounded draft with citations or "insufficient evidence" → MiniCheck-Flan-T5-Large claim verification → up to 2 rounds of targeted re-retrieval and revision when support < 0.1 → abstention by conformal risk control / Learn-then-Test on a logistic confidence score (support + token probability + closed-book confidence).

**Backbones.** Qwen2.5-7B-Instruct Q4_K_M (official GGUF, Apache-2.0) on llama.cpp b12110. Secondary: Llama-3.2-3B-Instruct Q4_K_M. Self-RAG 7B on its own backbone (community Q4_K_M GGUF, checked: reflection tokens intact).

**Baselines (12).** Closed-book; RAG top-5; RAG + same reranker; RAG top-10; CRAG (released evaluator, local index for web search); Chain-of-Note; FLARE; Self-RAG 7B; CoVe (factored); RARR; RAG + token-probability abstention; RAG + SelfCheck consistency.

**Data (test N).** PopQA long-tail 1,000 · TriviaQA 1,000 · HotpotQA 1,000 · PubHealth 987 · RGB 1,300 (noise 0/0.4/0.8, rejection, counterfactual) · RAGTruth QA 900 for the verifier alone. Self-RAG's released passages are used where they exist; otherwise a local 5.3 M-passage Wikipedia index (HotpotQA 2017 intros).

**Primary metrics.** Accuracy with abstention off (Self-RAG `match` rule), hallucination rate at 80% coverage, AURC. Paired bootstrap CIs, McNemar / permutation tests, Holm correction.

**Pilot (dev, 100 per dataset; means over 4 datasets).**

| | Accuracy | AURC ↓ | Hallucination @ 80% ↓ |
|---|---|---|---|
| **RAG-Verify** | 0.443 | 0.407 | 0.398 |
| Self-RAG 7B | **0.510** | 0.370 | **0.367** |
| RARR | 0.487 | 0.402 | 0.385 |
| RAG top-10 | 0.460 | **0.359** | 0.385 |
| RAG top-5 | 0.435 | 0.375 | 0.410 |

**On this dev evidence, RAG-Verify does not beat the strongest baselines.** The verifier's support score is a weak correctness signal for short answers (AUROC 0.42–0.71). Revision rarely changes an answer (10 of 100 on PopQA). The grounded prompt declines on about 40% of PopQA questions. With n = 100 these differences are within noise, but H1 does not look likely as designed.

**Compute.** Used 1.86 GPU-h. Projected total ≈ 32 of 50 GPU-h: main runs 20.3 h, SelfCheck seeds 0.9, secondary backbone ≈ 2, ablations ≈ 3.5, other ≈ 3. This assumes RARR uses one greedy question-generation round; the original 3 sampled rounds × 3 seeds would add about 15 h.

**Options.**
- **A** — freeze now and run. This is an honest test of H1–H4, but the paper would likely be an empirical study of when verification helps.
- **B (recommended)** — at most 2 more GPU-h of dev-only iteration on RAG-Verify (500-question dev sets), then freeze and run. Targets: drop or soften the "insufficient evidence" option, select evidence by verifier, a better short-answer claim format. The main runs still fit before T0 + 44 h.

**Top risks.**
1. H1 / H2 may fail; the results will be reported as they are.
2. Name and overlap with your ICTCS 2026 "RAG-Verify" paper: needs disclosure, possibly a new name (e.g., GVR-RAG; not yet collision-checked).
3. Close 2026 prior work (SURE-RAG, R2VC, PAVE, TARG): novelty is the combination plus the evaluation, not any single module (`literature/novelty.md`).
4. Provenance: Self-RAG eval files come from byte-identical HF mirrors (the Google Drive original refused download); the Self-RAG GGUF is a community conversion.
5. Laptop limits: the CPU reached 97 °C (fixed: retrieval models moved to the GPU, 2 CPU threads, load guard); VRAM spill on Windows (fixed with a leaner server profile). Long runs need AC power and clear vents.
6. PubHealth: with Wikipedia evidence the Qwen methods mostly answer "false" (dev accuracy 0.33–0.50; Self-RAG 0.69).
