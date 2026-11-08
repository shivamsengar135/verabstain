# PROTOCOL — RAG-Verify (pre-registration)

**Status: FROZEN 2026-11-05 (after the user's GO, option B).** Frozen copy: `logs/protocol_freeze/PROTOCOL.md`. Its SHA-256 then goes into
`COMPUTE_LEDGER.md`, and `scripts/run_queue.py` refuses test-split jobs until that hash matches. Later
changes go only into §17 (Deviations), dated and with reasons. Values that were marked as pilot-dependent have been filled in from the
Phase-2 dev pilot before the freeze. TARGET = A (text QA and fact verification; no video extension).

## 1. Research questions and hypotheses

| | Question | Pre-registered test |
|---|---|---|
| H1 | Does RAG-Verify have a lower hallucination rate than the baselines at matched coverage? | Primary: hallucination rate (answered-and-wrong ÷ all) at coverage 0.8, RAG-Verify vs each baseline, per core dataset; paired bootstrap CI + paired permutation test, Holm over baselines. H1 holds on a dataset only if RAG-Verify is significantly lower than **every** baseline there. Secondary: coverage 0.6/0.7/0.9 and AURC. |
| H2 | Does a small external verifier beat LLM self-verification, at lower cost? | RAG-Verify (M4 = external verifier) vs RAG-Verify (M4 replaced by the backbone judging support); AURC and hallucination at coverage 0.8 (bootstrap CI), cost = LLM calls, tokens and seconds per example. CoVe is reported as a published self-verification reference. |
| H3 | Does gating (M1) cut retrieval and LLM calls with ≤ 1 point accuracy loss? | RAG-Verify vs RAG-Verify without M1: accuracy difference with 95% CI; H3 holds if the CI lower bound is above −1 point **and** mean LLM calls and retrievals are lower. |
| H4 | Does realised risk on held-out data stay close to the target α? | §12: 100 random 50/50 calibration/evaluation splits per test pool, α ∈ {0.05, 0.10, 0.20}; report mean ± sd realised risk and the fraction of splits exceeding α (CRC: expected-risk guarantee; LTT: P(risk > α) ≤ δ = 0.1). |

## 2. Datasets, splits and sample sizes

Fixed evaluation order = `data/splits/<dataset>_<split>.ids` (seed-0 shuffle); a run with N uses the first N ids.

| Dataset | Test pool (source) | N test | Dev source (never test) | N dev | Passages for static methods | Primary metric |
|---|---|---|---|---|---|---|
| PopQA long-tail | 1,399 questions with s_pop < 100, Self-RAG `popqa_longtail_w_gs.jsonl` | 1,000 (first 1,000 of the fixed order) | 500 PopQA questions with s_pop ≥ 100 (PopQA has no official dev split; disjoint from test) | 100 pilot / 500 tuning | released list (first k) | Self-RAG `match` |
| TriviaQA | Self-RAG `triviaqa_test_w_gs.jsonl` (7,313) | 1,000 | 500 from TriviaQA unfiltered **train**, question text disjoint from test | 100 / 500 | released list | Self-RAG `match` |
| HotpotQA | distractor **validation** (7,405), answered open-domain over the local index | 1,000 | 500 from HotpotQA train | 100 / 500 | local index (§3) | Self-RAG `match` (EM, F1 secondary; chosen on dev because Self-RAG answers in sentences) |
| PubHealth | Self-RAG `health_claims_processed.jsonl` (987 true/false claims) | 987 | official PubHealth validation split, true/false labels only, 500 sampled, disjoint claims | 100 / 500 | released list | label accuracy |
| RGB (robustness / abstention) | `en.json` 300 questions × noise {0, 0.4, 0.8} + negative rejection (noise 1.0) + `en_fact.json` 100 counterfactual = 1,300 instances | all | none (no tuning on RGB) | — | 5 passages built by RGB's `processdata` (seed 2333) | RGB `checkanswer`; rejection rate; fake-answer rate |
| RAGTruth QA (component) | 900 test responses (QA task) | all | 600 train responses | 600 | source passages given | response-level detection AUROC, F1 |

Self-RAG's evaluation files were obtained from a Hugging Face mirror (`jesse-tong/selfrag-eval-data`) because the original Google Drive file refused automated download. Every file is byte-identical (LFS SHA-256) to a second, independent mirror (`awinml/crag_self_rag_datasets`). Recorded in `logs/downloads_manifest.tsv`; the user should check it once against the original if possible (TODO-VERIFY). LLM-AggreFact is not used: it is gated (needs an HF login) and its CC BY-ND licence is restrictive.

## 3. Corpus and retrieval

- **Local index** (`indexes/wiki17/`): BEIR HotpotQA corpus = HotpotQA's 2017-10-01 Wikipedia introductory paragraphs, 5,233,329 articles → 5,302,804 passages. Intros longer than 150 words are split into ≤ 100-word chunks (56,646 articles).
- **BM25**: own memory-mapped implementation (k1 = 0.9, b = 0.4, Lucene idf, lowercase `\w+`, scikit-learn English stop words, Snowball English stemmer); 2.55 M terms, 120 M postings.
- **Dense**: BAAI/bge-small-en-v1.5 (fp16, 384-d, query instruction prefix), FAISS IVF16384-SQ8, nprobe = 128.
- **Hybrid pool (M2 and the `+reranker` baseline)**: reciprocal rank fusion (k = 60) over the released list top-20 (if any), BM25 top-20 and dense top-20; the 30 best fused candidates are reranked with `cross-encoder/ms-marco-MiniLM-L6-v2`.
- **Static first-stage list** where no released passages exist (HotpotQA, all dev sets): **hybrid** RRF(BM25 top-20, dense top-20), chosen on dev by mean answer recall@5 (hybrid 0.649, dense 0.627, BM25 0.538; `analysis/retrieval_recall_dev.json`).
- **Follow-up queries** (FLARE, CRAG web step, CoVe, RARR, RAG-Verify M5) all use one shared `search()`: hybrid RRF(BM25, dense) → cross-encoder → top k_sub. On RGB the search runs over that question's own RGB passage pool.
- Every retrieval and rerank result is cached (`cache/retrieval.sqlite`).

## 4. Backbones and inference

- **Primary:** Qwen2.5-7B-Instruct, official GGUF Q4_K_M (`Qwen/Qwen2.5-7B-Instruct-GGUF` @ bb5d59e), Apache-2.0, ungated. It is widely used in 2024–2026 RAG work, its grouped-query KV cache is small enough for 8 parallel slots in 8 GB, and it has no thinking mode.
- **Secondary:** Llama-3.2-3B-Instruct Q4_K_M (`bartowski/Llama-3.2-3B-Instruct-GGUF` @ 5ab33fa; the official Meta repository is gated; Llama 3.2 Community Licence permits research use), on the subset in §14.
- **Self-RAG 7B:** official `selfrag/selfrag_llama2_7b` (MIT) as a community Q4_K_M GGUF (`QuantFactory/selfrag_llama2_7b-GGUF` @ 79baa8c). It has its own Llama-2-7B backbone and is labelled that way.
- **Serving:** llama.cpp `llama-server` build b12110, all layers on GPU, flash attention, q8_0 KV cache, 32,768 unified KV cells, 8 slots, continuous batching, prompt cache. Every model uses the same Q4_K_M quantization.
- **Decoding:** greedy (temperature 0) except sampled components (SelfCheck samples, RARR question generation), which use fixed seeds. Shared generation limits: final answer ≤ 48 tokens, grounded draft ≤ 96, notes/reasoning ≤ 384, plans/queries ≤ 96/32.
- **Determinism:** llama.cpp batching can introduce tiny numeric differences between identical requests. Every LLM call is cached (SQLite, key = SHA-256 of model tag + endpoint + full payload), so reported numbers are reproducible from the cache and the raw outputs.

## 5. RAG-Verify — exact algorithm

```
input: question/claim x; config C (one global setting, §7); verifier V
M1  d0, p(d0) ← closed-book greedy answer and geometric-mean token probability (same call as the closed-book baseline)
    if p(d0) ≥ C.gate_tau: return d0 (path=closed; score = p(d0))      [main configuration: gate_tau = 1.01, i.e. gate off — §7]
M2  P ← hybrid pool(x) reranked by cross-encoder; E ← top-k of P with CE logit ≥ C.rel_tau
M3  (a, s_text) ← grounded draft from E: answer only from E, cite passages, or "insufficient evidence"
    if a = insufficient evidence: a ← forced standard-RAG answer on E (same call as the +reranker baseline; flagged)
M4  claim ← template "The answer to the question \"x\" is a." (QA) or the input claim (fact verification); σ ← V(E, claim)
    for claims: support of the answer = σ if a = true else 1 − σ
M5  for r in 1..C.rounds while σ < C.sup_tau:
        q ← focused query from the backbone; E' ← search(q, k_sub) minus seen; E ← (E ∪ E')[:10]
        (a', s') ← revised grounded answer from E (keep a unless E supports a different answer)
        σ' ← V(E, s'); if σ' ≥ σ: (a, σ) ← (a', σ')
M6  score = logistic model on [logit σ, logit token-prob of final answer, logit p(d0), insufficient flag], fitted on the
    final dev runs devB2_ragverify_* (300 per dataset; chosen over σ alone and token-prob alone by 5-fold CV mean AURC
    0.382 vs 0.440 / 0.407); coefficients frozen in analysis/m6_score_model.json; abstain iff score < λ, λ from §12
```

The M4 verifier is chosen on dev only (RAGTruth dev detection AUROC plus support-vs-correctness AUROC on QA dev drafts) from MiniCheck-Flan-T5-Large, MiniCheck-DeBERTa-v3-Large and HHEM-2.1-Open. Dev result: RAGTruth AUROC 0.847 / 0.840 / 0.881; QA-draft support AUROC tied (MiniCheck-FT5 mean 0.569, HHEM 0.576). **Chosen: MiniCheck-Flan-T5-Large**, because HHEM truncates at 512 tokens while our evidence (5–10 passages) exceeds that; HHEM is run as a verifier-swap ablation. **Circularity guard:** the faithfulness evaluators (§8) come from model families different from the chosen M4 verifier.

## 6. Baselines

All baselines use the same backbone, quantization, released/first-stage passages, k, generation limits and dev-tuning procedure as RAG-Verify, except Self-RAG, which uses its own backbone.

| ID | Method | Source followed | Our adaptation (all recorded) |
|---|---|---|---|
| B1 | Closed-book | — | same answer prompt as every method |
| B2 | Standard RAG top-k (k = 5) | — | released passages where available |
| B3 | Standard RAG + reranker | — | same hybrid pool and cross-encoder as RAG-Verify, top-5 |
| B4 | Standard RAG large-k (k = 10) | — | first 10 static passages |
| B5 | CRAG | released code (`CRAG_Inference.py`, `internal/external_knowledge_preparation.py`, `utils.py`) and released T5-large evaluator | Google web search → our local `search()`; keyword prompt copied; 'excerption' decomposition; ambiguous = "Knowledge1 … [sep] Knowledge2 …" (the released combine script mis-uses `enumerate`, so we implement its evident intent); thresholds tuned on dev |
| B6 | Chain-of-Note | paper | prompting adaptation (original fine-tunes Llama-2): per-passage notes, then answer |
| B7 | FLARE (direct, implicit queries) | paper (§3: initial query = input, retrieve if any token prob < θ, masked queries) | short-form: the look-ahead sentence is the answer; query = question + answer tokens with prob ≥ β; ≤ 2 rounds; θ tuned on dev |
| B8 | Self-RAG 7B | released `run_short_form.py` | top-20 instead of 5,000 token probabilities; adaptive-retrieval rule chosen on dev (paper's normalised probability vs released code's log-probability ratio); README settings (threshold 0.2, max tokens 100 / 50, w = 1/1/0.5) |
| B9 | CoVe (factored) | paper | baseline answer = B2; ≤ 3 verification questions, each answered independently with its own retrieval (k_sub); final revised answer |
| B10 | RARR | released prompts and pipeline (`run_editor_sequential.py`) | edits the B2 answer as a statement with the question as context; local `search()` → 1 evidence per question; ≤ 3 questions; final extraction call |
| A1 | B2 + token-probability abstention | — | B2's outputs scored by answer token probability |
| A2 | B2 + SelfCheckGPT-style consistency | SelfCheckGPT (sampling consistency) | 5 samples at T = 1.0, score = agreement with the greedy answer |

**Excluded for feasibility, decided before any test result:** DRAGIN (needs attention weights; not available through llama.cpp within budget); Adaptive-RAG (needs its trained complexity classifier); Verify-and-Edit (RARR represents post-hoc revision); 2025–2026 methods without public code or trained for other settings (R2VC, PAVE, RH-RAG); Astute RAG / InstructRAG-ICL (not included: the projection leaves no room for a 13th method once RARR alone costs ~6 s per item).

## 7. Dev tuning (equal budget for every method)

Each thresholded method gets one "maximal" dev run that records every branch: RAG-Verify with the gate off and all revision rounds; FLARE with θ = 1; CRAG computing all three actions; Self-RAG computing both paths. Thresholds are then simulated offline on fixed grids (`analysis/tune_dev.py`). The single global setting that maximises the mean primary dev accuracy over the four dev datasets is chosen; ties go to the cheaper setting. RAG-Verify's gate is then set to the most aggressive value whose dev accuracy is within 1 point of gate-off **on every dev dataset** (changed on 2026-11-05 before the freeze from "on the mean": the mean was dominated by PubHealth, where closed-book beats retrieval, and hid an 11-point HotpotQA loss). No gate value satisfied this, so the main configuration has the gate off and H3 is tested in the ablation with gate_tau = 0.9 (the value that maximised mean dev accuracy). Prompts are fixed before the pilot and are not tuned per method beyond this. No test data is used.

## 8. Metrics

- **Primary:** (i) accuracy with abstention off (rule per §2; a method's own "unknown" counts as wrong except for RGB rejection); (ii) hallucination rate at coverage 0.8; (iii) AURC. Coverage and AURC use each method's pre-specified confidence score: final-answer geometric-mean token probability (B1–B7, B9, B10, A1), Self-RAG's path score (B8), sample agreement (A2), the frozen logistic score of §5 (RAG-Verify).
- **Secondary:** EM, F1, normalised match; selective accuracy; native abstention rate; RGB accuracy per noise rate, rejection rate (noise 1.0) and fake-answer rate (counterfactual); verifier AUROC/F1 on RAGTruth.
- **Faithfulness:** unsupported-answer rate = fraction of answered QA items whose answer statement is not supported (score < 0.5) by its cited/used evidence. It is scored by two evaluators from families different from M4: AlignScore-large (RoBERTa) and DeBERTa-v3-large NLI (if M4 = MiniCheck-Flan-T5).
- **Efficiency:** LLM calls, prompt and completion tokens, retrievals, verifier pairs, wall seconds per example, GPU-seconds per run, peak VRAM.
- **No LLM-as-judge.**

## 9. Statistical analysis

- Paired bootstrap 95% CIs (10,000 resamples, seed 0) for every RAG-Verify − baseline difference.
- McNemar exact test for accuracy; paired permutation test (10,000) for hallucination indicators at matched coverage; bootstrap for AURC.
- Holm–Bonferroni across the 12 baseline comparisons (B1–B10, A1, A2) within each dataset × primary metric.
- Effect sizes are reported as point differences with CIs and N.
- Sampled components (A2, RARR question generation) use 3 seeds and report mean ± sd.

## 10. Ablations (Phase 4; PopQA-LT and HotpotQA, first 500 test items each)

−M1 (gate off), −M2 (static top-k evidence), −M3 (standard RAG prompt), −M4 (no verification ⇒ no M5; score = token probability), −M5 (verify without revision), −M6 (no abstention; affects only risk–coverage), M4 → self-verification (H2), verifier swap (the other two candidates), k ∈ {3, 5, 10}, rounds ∈ {0, 1, 2}, secondary backbone, and a quantization check (Llama-3.2-3B Q4_K_M vs Q8_0 on 200 examples).

## 11. Robustness

RGB: noise 0 / 0.4 / 0.8, negative rejection (no relevant evidence) and counterfactual passages. If budget allows: a HotpotQA distractor experiment on 300 test questions, with 0/25/50/75% of the 5 passages replaced by HotpotQA's distractor paragraphs.

## 12. Risk-control evaluation (M6)

For each test pool and α ∈ {0.05, 0.10, 0.20}: 100 random 50/50 splits (seeds 0–99).

- **Conformal risk control** sets λ on the calibration half for L(λ) = 1{score ≥ λ}·1{wrong} (overall hallucination rate; B = 1) using the rule n/(n+1)·R̂ + 1/(n+1) ≤ α.
- **Learn-then-Test** (fixed-sequence binomial tests, δ = 0.1) controls error among answered questions.
- We report realised risk on the evaluation half (mean ± sd), coverage, and the fraction of splits with risk > α.
- **Assumption:** calibration and evaluation items are exchangeable (random splits of one pool). Under distribution shift (e.g., calibrating on one dataset and applying to another) no guarantee is claimed; this is reported as a secondary analysis.

## 13. Efficiency analysis

Cost-quality Pareto plots (accuracy and hallucination at coverage 0.8 vs LLM calls and seconds per example).

## 14. Compute plan (measured in the pilot)

Pilot wall-clock cost per example with 8 parallel slots (s): closed-book 0.05, RAG top-k 0.34, +reranker 0.62,
large-k 0.48, SelfCheck 0.41, Chain-of-Note 1.28, FLARE 0.54, CoVe 1.38, CRAG 1.97, RARR 6.15, Self-RAG 0.90,
RAG-Verify 2.37 (pilot "maximal" settings; tuned settings are cheaper). GPU-hours used through the pilot: 1.86.

| Block | Size | Projected GPU-h |
|---|---|---|
| Main: 12 methods + RAG-Verify, primary backbone | 5,287 test items (4 × ~1,000 + RGB 1,300) | 20.3 (RARR with one greedy question-generation round, see §17 proposal) |
| Extra seeds (SelfCheck × 2) | same | 0.9 |
| Secondary backbone (Llama-3.2-3B) | PopQA, HotpotQA, PubHealth × 500; closed-book, RAG top-k, +reranker, CRAG, FLARE, CoVe, A1/A2, RAG-Verify | ≈ 2 |
| Ablations | PopQA + HotpotQA × 500; −M1…−M5, self-verification, HHEM swap, k ∈ {3, 10}, rounds ∈ {0, 1} | ≈ 3.5 |
| Robustness extra (HotpotQA distractors), quantization check, RAGTruth test, dev confirmation run | | ≈ 3 |
| **Total projected (incl. pilot)** | | **≈ 32 of 50** | Per-job projections are logged before launch. The runner refuses jobs that would end after 2026-11-07 15:46 IST.

## 15. Scope-cut order (logged before seeing affected results)

1. Optional long-form and TARGET-B extras (none planned).
2. Secondary-backbone extras beyond the minimal subset.
3. Ablations on fewer datasets.
4. Smaller N per dataset (never below 500 for main datasets).

## 16. Reporting

Every method, dataset and metric above appears in the paper, including losses and ties. Published numbers appear only in a separate table (`literature/reported_results.csv`).

## 16b. Final dev readout before the freeze (dev only; 300 per dataset; means over 4 dev sets)

Accuracy / AURC / hallucination at coverage 0.8 — RAG-Verify 0.450 / 0.382 / 0.383; RAG top-10 0.473 / 0.366 / 0.376;
Self-RAG 0.488 / 0.408 / 0.384; CRAG 0.461 / 0.366 / 0.388; FLARE 0.466 / 0.372 / 0.386 (RAG-Verify AURC is 5-fold
cross-validated). On dev, RAG-Verify is competitive but not better; this was recorded before any test run.

## 17. Deviations

(none yet — the protocol is not frozen. Pre-freeze decisions taken on dev are recorded inline: match rule primary
for HotpotQA; hybrid static retriever; MiniCheck-FT5 verifier; logistic M6 score; dev-tuned thresholds in
`analysis/dev_tuning.json`: CRAG upper 2.0 / lower −3.0, FLARE θ 0.8, Self-RAG threshold 0 (always retrieve),
RAG-Verify gate off (see §7), support threshold 0.1, fallback = forced RAG answer, M4 claim = template statement
(dev AUROC of support vs correctness: template 0.778 / 0.504 / 0.614 vs model statement 0.713 / 0.420 / 0.574 on
PopQA / TriviaQA / HotpotQA dev drafts). RARR uses one greedy question-generation round instead of three sampled
rounds (budget; accepted at Gate 1, 2026-11-05) and is therefore deterministic (no extra seeds).)

**2026-11-05 13:50 IST — Phase-4 specification details (fixed before any Phase-4 run; no test result of an ablation,
secondary-backbone or quantization run existed).** (a) "−M1 (gate off)" is the main configuration itself, because the
gate is off there (§7); the H3 contrast therefore runs RAG-Verify *with* M1 at gate_tau = 0.9 on the same 500 items.
(b) rounds = 0 is behaviourally identical to −M5 and is reported from the −M5 run instead of being run twice.
(c) Ablations use the first 500 items of the main PopQA-LT / HotpotQA test order; the main-configuration reference for
them is the same 500-item prefix of the main runs. (d) The secondary backbone runs closed-book, RAG top-k (+A1 offline),
+reranker, FLARE, CoVe, SelfCheck (A2, seed 0 only), CRAG and RAG-Verify, with all thresholds as frozen (no re-tuning
for Llama-3.2-3B). (e) The quantization check compares Llama-3.2-3B Q8_0 against Q4_K_M with closed-book, RAG top-k and
RAG-Verify on the first 200 items of PopQA-LT and HotpotQA (Q4 numbers = the 200-item prefix of the secondary-backbone
runs). Queue: `jobs/queue_phase4.yaml` (52 jobs, generated by `scripts/make_queue.py phase4`).

**2026-11-05 15:10 IST — HotpotQA distractor experiment specification (§11; fixed before any run of it).** Items: the
first 300 of the fixed HotpotQA test order that have ≥ 4 distractor paragraphs (3 of the first 303 have fewer).
Base context: the 5 passages B2 received in the main run. Levels 0/25/50/75%: m = ⌊p·5 + 0.5⌋ = 0/1/3/4 lowest-ranked
base passages replaced by the question's first m distractor paragraphs. Every method sees only these 5 passages (they are
also the search pool, as for RGB), so M5 and the CRAG/FLARE-style follow-up searches cannot add new passages. Methods:
RAG top-k, +reranker, Chain-of-Note, CRAG, Self-RAG and RAG-Verify (24 jobs, `jobs/queue_distractor.yaml`; data built
by `scripts/make_distractor_data.py`). Runs after Phase 4 and is the first block cut if the time budget runs short (§15).

**2026-11-06 22:03 IST — RGB counterfactual scoring bug fix (scoring only; no run affected).** `scripts/prepare_data.py`
stored RGB `en_fact` answers as `[inst["answer"]]`. For the 28 of 100 counterfactual items whose original answer is
already a list of answer variants, this added one nesting level that RGB's `checkanswer` does not expect (the first
scoring pass crashed). `scripts/score.py` now unwraps this extra level for counterfactual items, so the ground truth equals
RGB's `instance['answer']` exactly as in the reference `evalue.py`. Gold answers are never shown to any method, so all
outputs are unchanged. Every run was scored after the fix, and no RGB counterfactual score had been computed before it.

**2026-11-06 22:07 IST — LTT implementation bug fix (analysis only; found from test output).** As first implemented,
`ltt_threshold` tested thresholds from strictest to most lenient and stopped at the first non-rejection. The strictest
threshold answers one calibration item, and H0 cannot be rejected there even with zero errors ((1−α)^n > δ). LTT
therefore certified no threshold, and coverage was 0 on every test pool. This came to light in the first test-set run of
`analysis/risk_control.py`, which is how it was found. Fix: thresholds where rejection is impossible
((1−α)^n > δ, with n the number of answered calibration items) are skipped. Skipping depends only on the calibration
scores, never on the correctness labels, so the testing order stays fixed given the scores. The procedure is otherwise
unchanged (binomial p-values, δ = 0.1, 200-point quantile grid). The CRC results are unaffected. LTT was re-run on all
pools after the fix.

**2026-11-06 23:00 IST — Phase-4 queue naming bug (no result affected).** The two MiniCheck-DeBERTa ablation jobs
named the verifier `minicheck-deb-l` (the model folder) instead of its registry name `minicheck-deb`. The PopQA-LT job
failed at verifier construction (22:12) before producing any output. The queue and `scripts/make_queue.py` now use
`minicheck-deb`, which is the same model (`models/minicheck-deb-l`). Both jobs run in the watchdog's next pass over
unfinished Phase-4 jobs.

**2026-11-07 10:47 IST — Post-hoc analyses after an internal pre-submission review (decided before any of their results exist; reported
separately from the pre-registered results and labelled post-hoc).** A simulated review (reviews/simulated_review_round0.md)
raised three fairness questions. The following analyses are added; none changes any pre-registered result or decision rule.
(P1) *Equal evidence for corrective baselines.* Chain-of-Note and CRAG are re-run with the same hybrid pool and cross-encoder
top-5 as RAG-Verify's M2 (`method_args: {evidence: m2}`; identical to B3's evidence because rel_tau is off), on the first
500 test items of PopQA-LT, TriviaQA, HotpotQA and PubHealth (8 jobs, `jobs/queue_posthoc.yaml`). RGB is excluded because
every method there already receives the same 5 RGB passages. Comparison: RAG-Verify minus each re-run baseline on the same
500 items, paired bootstrap CI, McNemar / paired permutation, Holm over the 2 methods within dataset × metric.
Budget: 44.8 of 50 GPU-h were used before these runs; the runner's 15:46 cut-off and thermal guard apply.
(P2) *Equal confidence-score fitting.* Every method, RAG-Verify included, gets a logistic confidence score fitted the
same way: features = all per-example signals the method itself produces (token probabilities, and for RAG-Verify
also support, c1 and the insufficient-evidence flag; Self-RAG its path score; A2 its agreement), two-fold
cross-fitting on the test outputs (random 50/50 split, seed 0; fit on one half, score the other, then swap).
AURC and H@0.8 are recomputed with these scores and compared with paired bootstrap CIs; CPU only, from existing
outputs.
(P3) *Paired faithfulness.* The two faithfulness judges are re-run storing per-item scores for RAG-Verify, B3, B5, B6 and B8,
and unsupported rates are compared on the items answered by both methods, judged against each method's own evidence
(GPU, run only if P1 finishes with budget and time left).

**2026-11-07 13:50 IST — Follow-up study "RAG-Verify v2" (user decision; registered before any v2 development or fresh-item
run).** The user asked for further ablations and an improved method. Because every test item used so far has been seen,
the improved method is developed and evaluated as a separate follow-up study; it never replaces any result above and is
reported in full whatever its outcome. Its design ideas are informed by the pre-registered results (e.g. the M5 ablation
and the PubHealth loss); this is disclosed.
(V1) *Development data.* v2 candidates are chosen only on dev items 0–299 of PopQA-LT, TriviaQA, HotpotQA and PubHealth
(the items already used for v1 tuning and for the dev baseline runs devB_*/pilot_*). No test output is used for any v2
choice.
(V2) *Fresh test sets, fixed now.* PopQA-LT test items 1000–1398 (399; the remainder of the test split), TriviaQA test
items 1000–1499 (500), HotpotQA test items 1000–1499 (500), and PubHealth dev items 300–499 (200; never run by any job and
not used by any tuning script — the PubHealth test split is exhausted). Positions refer to data/splits/*.ids. RGB is
excluded: it has no unused items. These items are touched only after the v2 freeze below.
(V3) *Methods on the fresh items.* RAG-Verify v2, RAG-Verify v1 (main configuration, unchanged), and all 12 baselines
with their frozen configurations. Decided now for feasibility: A2 and RARR are run with seed 0 only (the main study used
3 seeds).
(V4) *Analysis.* As in §9: accuracy, H@0.8 and AURC; v2 minus each baseline with paired bootstrap 95% CIs, McNemar /
paired permutation tests and Holm over the 12 baselines within dataset × metric; v2 minus v1 reported separately. v2's
M6 score is fitted on dev items 0–299 only, then frozen. CRC/LTT are not re-evaluated.
(V5) *Freeze.* The v2 configuration and code are recorded in a further dated entry, and the SHA-256 of PROTOCOL.md is
written to COMPUTE_LEDGER.md, before any fresh-item run. Fresh-item runs are logged in logs/test_runs.log.
(V6) *Budget.* By user decision the time limit is extended by 24 h (no new GPU jobs after 2026-11-08 15:46, hard stop
2026-11-08 21:46) and the GPU budget by 12 h (62 GPU-h total). The thermal guard is unchanged.

**2026-11-07 23:15 IST — Freeze of RAG-Verify v2 (V5; before any fresh-item run).** Method `ragverify_v2`
(src/ragverify/methods.py, class RAGVerifyV2), frozen models analysis/v2/v2_model.json (fitted by analysis/v2/fit_v2.py on
dev items 0–299 only). QA: M2 evidence (top-5 of the hybrid pool after cross-encoder reranking); three candidates — the
grounded draft D (standard-RAG answer if the draft says "insufficient evidence"), the standard RAG answer R on the same
evidence, and the closed-book answer C; MiniCheck-FT5 support of each candidate's template statement; a logistic selector
(features: logit support, logit c1 for C, candidate type, number of other candidates agreeing, agreement with C × logit c1,
draft-insufficient flag) picks the answer; its probability is the confidence score. No gate, no revision. Claims: label
"true" iff the logistic rule on logit MiniCheck support of the claim gives P(true) ≥ 0.5 (no LLM call; in effect "true"
unless the support is below ≈0.006, i.e. mostly the dev class prior plus a weak signal); confidence = max(P, 1−P).
Dev readout (items 0–299; in-sample for the fitted models; 5-fold CV of the selector in parentheses): accuracy PopQA dev
0.440 (0.440), TriviaQA 0.520 (0.517), HotpotQA 0.497 (0.497), PubHealth 0.647 (claim rule CV 0.647); v1 on the same items
0.427 / 0.480 / 0.483 / 0.410. Dev evidence for selective prediction was mixed (CV AURC better on TriviaQA, worse on PopQA
and HotpotQA than v1's in-sample M6). Analysis script fixed now: analysis/v2/fresh_stats.py. Queue: jobs/queue_fresh.yaml
(52 jobs; v2 and v1 first; then the baselines; projected 13.1 GPU-h with a 1.3 safety factor). If the 15:46 cut-off stops
the queue, the unfinished methods are reported as not run.

**2026-11-08 08:35 IST — v2 queue timing (no design change).** The fresh queue was paused 2026-11-07 23:19 at the user's
request (laptop shutdown) and resumed 2026-11-08 08:29 with the per-item outputs kept. By user decision the time limit is
extended by a further 12 h (no new GPU jobs after 2026-11-09 03:46). Methods, items, models and analysis are unchanged.

**2026-11-08 18:40 IST — Follow-up study v2: completion and reporting (no design change).** All 52 fresh-item jobs finished at 17:56 (no cut-off reached). Analysis ran with the frozen analysis/v2/fresh_stats.py; after the results were known, only its table writer was changed: LaTeX escaping fixed (the frozen version wrote control characters), a "v1 − best baseline" row added to table_fresh.tex, and a full every-method table (table_fresh_full.tex) added. No statistic, item, method or test changed. Reported additionally (descriptive, computed by analysis/make_paper_numbers.py): on the PubHealth fresh items v2 labelled 199 of 200 claims "true" against a majority-class rate of 0.625, so its accuracy gain over v1 reflects the class prior. analysis/make_paper_numbers.py now loads fontTools' pure-Python bezierTools when Windows Application Control blocks the compiled module (figure rendering only; no number affected).
