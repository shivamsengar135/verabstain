# Novelty assessment — RAG-Verify (Phase 1, 2026-11-04)

Every claim about another paper below was checked only against its arXiv abstract (stored in
`literature/meta/<key>.json`). Full-text checks are still required before anything here goes into the
manuscript (TODO-VERIFY).

## Search log

| Date (IST) | Tool | Query | Notable hits |
|---|---|---|---|
| 11-04 22:05 | web (extended) | "RAG-Verify" retrieval-augmented generation verification | GitHub `shivamsengar135/RAG-Verify` (ICTCS 2026 submission); VG-RAG (Sci. Rep. 2026); "Reason and Verify" (2603.10143) |
| 11-04 22:05 | web | conformal risk control RAG abstention hallucination guarantee | C-RAG, UniCR (2509.01455, **withdrawn by arXiv**), 2512.15068, 2602.01637, CiteGuard (ICML 2026) |
| 11-04 22:05 | web | claim-level verification + targeted re-retrieval + revision + abstention | RH-RAG (2608.01311), R2VC (2609.11955), repair study (2608.29307), PAVE (2603.20673) |
| 11-04 22:05 | web | 2025 robust RAG beating CRAG/Self-RAG on PopQA | RPO (2501.13726), Open-RAG |
| 11-04 23:28 | web | external NLI verifier + conformal abstention + revision/re-retrieval | conformal abstention (2405.01563), 2512.15068, conformalised retrieval (2605.05244) |
| 11-04 23:28 | web | uncertainty gate + claim verification (MiniCheck) + revise/abstain, small open models | SURE-RAG (2605.03534), TARG (2511.09803), uncertainty-aware fact-checking (2604.11036), 2603.28444 |

Limitations of this search: six web searches plus targeted lookups. It is not a systematic review. Google
Scholar, DBLP (bot-blocked for scripts) and ACL Anthology full-text search were not queried.

## Name check

- No published *method* named "RAG-Verify" was found.
- The name is used by the GitHub repository `shivamsengar135/RAG-Verify`: "When Does Claim-Level Verification Help Small Retrieval-Augmented LLMs? An Empirical Study with RAG-Verify". Its README describes it as an ICTCS 2026 (Springer) submission with Qwen2.5-7B / Llama-3.1-8B (Ollama) on an RTX 4060, HotpotQA, TriviaQA and HaluEval-QA.
- **This looks like the user's own prior work. User decision needed.** Possible consequences: (a) the journal paper must disclose the overlap and cite or describe the ICTCS paper; (b) a journal extension usually needs substantially new content; (c) reusing the name ties the two papers together.
- Alternative names, not yet checked for collisions (TODO-VERIFY): **GVR-RAG** (Gated, Verified, Risk-controlled RAG), **VeriGate-RAG**, **CRV-RAG** (Calibrated Retrieve-Verify-Revise).

## Component-by-component prior art (none of the six modules is new on its own)

| Module | Closest prior work (from abstracts) |
|---|---|
| M1 uncertainty-gated retrieval | FLARE (token-probability trigger); Rowen (consistency-based trigger for hallucination mitigation); SKR, SeaKR, Adaptive-RAG, DRAGIN; TARG (2511.09803, training-free gate from a no-context draft); Moskvoretskii et al. 2025 (uncertainty methods for adaptive retrieval) |
| M2 hybrid retrieval + reranking | reciprocal rank fusion (Cormack et al., 2009); cross-encoder reranking (Nogueira & Cho, 2019); R2VC uses hybrid sparse+dense retrieval |
| M3 grounded drafting with citations / abstain option | ALCE (citations); Chain-of-Note ("unknown"); RGB negative rejection |
| M4 claim-level external verification with a small model | MiniCheck (the verifier itself); FActScore / SAFE (atomic claims); RH-RAG (NLI checker); PAVE (premise support scores); SURE-RAG (claim–evidence verifier + uncertainty); R2VC (NLI cross-encoder); RAGTruth repair study |
| M5 targeted re-retrieval + minimal revision | RARR (research-and-revise); EVER (real-time verify-and-rectify); Verify-and-Edit; CRITIC; RH-RAG (attestation-driven revision); PAVE (support-gated revision); RAGTruth repair study (delete/replace/rewrite) |
| M6 risk-controlled abstention | conformal abstention (Yadkori et al., 2024, self-consistency score, closed-book); C-RAG (certified RAG generation risk); TRAQ (conformal prediction sets for RAG QA); conformal factuality (Mohri & Hashimoto, 2024); chance-constrained inference (TMLR 2026); SURE-RAG and R2VC (threshold / calibrated abstention, not conformal per abstract); 2512.15068 reports that conformal NLI/embedding detectors fail on real hallucinations |

## What may be new (to be stated cautiously; no "first to" claim)

1. **The combination.** From the abstracts, no single system had all of: (i) a closed-book uncertainty gate, (ii) an external small-model claim verifier whose support score both triggers targeted re-retrieval and minimal revision **and** (iii) serves as the score for distribution-free risk control (CRC / LTT) of the hallucination rate.
   - Nearest: R2VC (verification + calibrated abstention, no revision loop, no conformal guarantee per abstract, fine-tuned generator); SURE-RAG (verification + threshold abstention); PAVE / RH-RAG (verification + revision, no risk control); conformal abstention (risk control, no retrieval-side verification).
   - A combination of known parts is a modest contribution.
2. **The evaluation design** may be the stronger contribution:
   - a compute-matched comparison against 10 re-implemented baselines on one open 7B backbone on an 8 GB laptop GPU;
   - matched-coverage hallucination comparisons;
   - explicit tests of H1–H4, including whether a small verifier beats LLM self-verification at lower cost (H2);
   - realised-risk checks for conformal abstention on RAG (H4).
3. Whether this adds anything beyond the user's ICTCS 2026 paper depends on that paper's content, which I have not read. **User input needed.**

## Risks to novelty

- 2026 preprints (SURE-RAG, R2VC, PAVE, TARG) overlap substantially; reviewers will expect comparisons or discussion. Most have no public code (checked only via abstracts; TODO-VERIFY).
- 2512.15068 reports negative results for conformal NLI-based detection on real hallucinations, which is directly relevant to H4.
