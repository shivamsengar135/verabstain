# Compute ledger — RAG-Verify

- **T0:** 2026-11-04 21:46:34 +05:30 (IST)
- **Hard stop (T0+72h):** 2026-11-07 21:46:34 +05:30 — all GPU work stops.
- **No new GPU jobs after (T0+66h):** 2026-11-07 15:46:34 +05:30, except bug re-runs. `scripts/run_queue.py` refuses any job whose projected end passes this time.
- **GPU experiment budget:** ≤ 50 GPU-hours in total, projected from measured pilot throughput.
- Window targets: Gate 1 by 2026-11-05 05:46 · main runs until 2026-11-06 17:46 · ablations/robustness until 2026-11-07 09:46.

## Protocol freeze

- PROTOCOL.md SHA-256: 61bf6b307abffbd41b6d25b0a4742c274dbdce61acaa1943646b849bad9f104c
- Frozen at: 2026-11-05 03:16:41 +05:30 (copy: logs/protocol_freeze/PROTOCOL.md; git commit follows)

## Budget summary (updated by hand at phase ends; per-job rows below are written by the runner)

| Category | Projected GPU-h | Used GPU-h |
|---|---|---|
| Phase 0 setup / benchmark | — | — |
| Indexing / retrieval caching | — | — |
| Phase 2 pilot (cap 4 h) | — | — |
| Phase 3 main runs | — | — |
| Phase 4 ablations / robustness / efficiency | — | — |
| **Total (cap 50 h)** | — | — |

## Job log

Columns: run_id · launched (IST) · projected runtime · projected end · wall time · GPU-seconds · LLM calls · prompt tokens · completion tokens · status · notes

| run_id | launched | projected | proj. end | wall | GPU-s | LLM calls | prompt tok | compl. tok | status | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| smoke_cb_tqa | 2026-11-04T22:47 | 2 min | 2026-11-04T22:49 | 0.0 min | 2 | 5 | 335 | 34 | complete |  |
| smoke_rag_tqa | 2026-11-04T22:47 | 2 min | 2026-11-04T22:49 | 0.2 min | 15 | 5 | 3093 | 24 | complete |  |
| smoke_con_pub | 2026-11-04T22:47 | 2 min | 2026-11-04T22:49 | 0.2 min | 14 | 5 | 3728 | 26 | complete |  |
| smoke_flare_hot | 2026-11-04T22:48 | 2 min | 2026-11-04T22:50 | 0.4 min | 27 | 5 | 3079 | 27 | partial 4/5 |  |
| smoke_selfcheck_pop | 2026-11-04T22:48 | 2 min | 2026-11-04T22:50 | 0.3 min | 18 | 30 | 13056 | 203 | complete |  |
| smoke_con_pub | 2026-11-04T22:49 | 2 min | 2026-11-04T22:51 | 0.2 min | 13 | 5 | 3768 | 878 | complete |  |
| pilot_closed_book_popqa_lt | 2026-11-04T23:24 | 10 min | 2026-11-04T23:34 | 0.1 min | 6 | 100 | 5215 | 555 | complete |  |
| pilot_closed_book_triviaqa | 2026-11-04T23:24 | 10 min | 2026-11-04T23:34 | 0.1 min | 5 | 95 | 5980 | 433 | complete |  |
| pilot_closed_book_hotpotqa | 2026-11-04T23:25 | 10 min | 2026-11-04T23:35 | 0.1 min | 5 | 100 | 7179 | 508 | complete |  |
| pilot_closed_book_pubhealth | 2026-11-04T23:25 | 10 min | 2026-11-04T23:35 | 0.0 min | 3 | 100 | 7140 | 200 | complete |  |
| pilot_rag_topk_popqa_lt | 2026-11-04T23:25 | 10 min | 2026-11-04T23:35 | 0.4 min | 23 | 100 | 51203 | 463 | complete |  |
| pilot_rag_topk_triviaqa | 2026-11-04T23:25 | 10 min | 2026-11-04T23:35 | 0.5 min | 27 | 100 | 60707 | 494 | complete |  |
| pilot_rag_topk_hotpotqa | 2026-11-04T23:25 | 10 min | 2026-11-04T23:35 | 0.5 min | 28 | 100 | 61437 | 495 | complete |  |
| pilot_rag_topk_pubhealth | 2026-11-04T23:26 | 10 min | 2026-11-04T23:36 | 1.0 min | 61 | 100 | 61769 | 200 | complete |  |
| pilot_rag_rerank_popqa_lt | 2026-11-04T23:27 | 10 min | 2026-11-04T23:37 | 1.1 min | 68 | 100 | 49361 | 441 | complete |  |
| pilot_rag_rerank_triviaqa | 2026-11-04T23:28 | 10 min | 2026-11-04T23:38 | 1.3 min | 78 | 100 | 59017 | 456 | complete |  |
| pilot_rag_rerank_hotpotqa | 2026-11-04T23:30 | 10 min | 2026-11-04T23:40 | 1.3 min | 79 | 80 | 47673 | 404 | complete |  |
| pilot_rag_rerank_pubhealth | 2026-11-04T23:34 | 10 min | 2026-11-04T23:44 | 0.4 min | 24 | 51 | 29436 | 102 | complete |  |
| pilot_rag_largek_popqa_lt | 2026-11-04T23:34 | 10 min | 2026-11-04T23:44 | 0.7 min | 42 | 100 | 96700 | 444 | complete |  |
| pilot_rag_largek_triviaqa | 2026-11-04T23:35 | 10 min | 2026-11-04T23:45 | 0.8 min | 49 | 100 | 112560 | 474 | complete |  |
| pilot_rag_largek_hotpotqa | 2026-11-04T23:36 | 10 min | 2026-11-04T23:46 | 0.8 min | 51 | 100 | 114109 | 498 | complete |  |
| pilot_rag_largek_pubhealth | 2026-11-04T23:37 | 10 min | 2026-11-04T23:47 | 0.8 min | 49 | 100 | 115584 | 200 | complete |  |
| pilot_rag_selfcheck_popqa_lt | 2026-11-04T23:38 | 10 min | 2026-11-04T23:48 | 0.6 min | 39 | 500 | 256015 | 2426 | complete |  |
| pilot_rag_selfcheck_triviaqa | 2026-11-04T23:38 | 10 min | 2026-11-04T23:48 | 0.7 min | 44 | 500 | 303535 | 2414 | complete |  |
| pilot_rag_selfcheck_hotpotqa | 2026-11-04T23:39 | 10 min | 2026-11-04T23:49 | 0.7 min | 45 | 500 | 307185 | 2638 | complete |  |
| pilot_rag_selfcheck_pubhealth | 2026-11-04T23:40 | 10 min | 2026-11-04T23:50 | 0.6 min | 35 | 500 | 308845 | 1000 | complete |  |
| pilot_chain_of_note_popqa_lt | 2026-11-04T23:40 | 10 min | 2026-11-04T23:50 | 1.9 min | 117 | 100 | 61103 | 15913 | complete |  |
| pilot_chain_of_note_triviaqa | 2026-11-04T23:42 | 10 min | 2026-11-04T23:52 | 2.1 min | 128 | 100 | 70607 | 16640 | complete |  |
| pilot_chain_of_note_hotpotqa | 2026-11-04T23:44 | 10 min | 2026-11-04T23:54 | 2.1 min | 125 | 100 | 71337 | 16506 | complete |  |
| pilot_chain_of_note_pubhealth | 2026-11-04T23:46 | 10 min | 2026-11-04T23:56 | 2.4 min | 142 | 100 | 71469 | 19430 | complete |  |
| pilot_flare_popqa_lt | 2026-11-04T23:49 | 10 min | 2026-11-04T23:59 | 0.9 min | 56 | 137 | 70369 | 687 | complete |  |
| pilot_flare_triviaqa | 2026-11-04T23:50 | 10 min | 2026-11-05T00:00 | 1.0 min | 57 | 123 | 73659 | 669 | complete |  |
| pilot_flare_hotpotqa | 2026-11-04T23:51 | 10 min | 2026-11-05T00:01 | 1.0 min | 59 | 115 | 68943 | 592 | complete |  |
| pilot_flare_pubhealth | 2026-11-04T23:52 | 10 min | 2026-11-05T00:02 | 0.7 min | 43 | 89 | 52839 | 178 | complete |  |
| pilot_cove_popqa_lt | 2026-11-04T23:52 | 10 min | 2026-11-05T00:02 | 2.1 min | 128 | 500 | 130986 | 11850 | complete |  |
| pilot_cove_triviaqa | 2026-11-04T23:55 | 10 min | 2026-11-05T00:05 | 2.2 min | 134 | 499 | 145499 | 13179 | complete |  |
| pilot_cove_hotpotqa | 2026-11-04T23:57 | 10 min | 2026-11-05T00:07 | 2.3 min | 137 | 493 | 138810 | 13253 | complete |  |
| pilot_cove_pubhealth | 2026-11-04T23:59 | 10 min | 2026-11-05T00:09 | 2.5 min | 152 | 500 | 147109 | 15443 | complete |  |
| pilot_rarr_popqa_lt | 2026-11-05T00:02 | 10 min | 2026-11-05T00:12 | 0.0 min | 0 | 0 | 0 | 0 | partial 0/100 |  |
| pilot_rarr_triviaqa | 2026-11-05T00:02 | 10 min | 2026-11-05T00:12 | 0.0 min | 0 | 0 | 0 | 0 | partial 0/100 |  |
| pilot_rarr_hotpotqa | 2026-11-05T00:02 | 10 min | 2026-11-05T00:12 | 0.0 min | 0 | 0 | 0 | 0 | partial 0/100 |  |
| pilot_rarr_pubhealth | 2026-11-05T00:02 | 10 min | 2026-11-05T00:12 | 0.0 min | 0 | 0 | 0 | 0 | partial 0/100 |  |
| pilot_ragverify_popqa_lt | 2026-11-05T00:02 | 15 min | 2026-11-05T00:17 | 2.6 min | 157 | 418 | 248169 | 11023 | complete |  |
| pilot_ragverify_triviaqa | 2026-11-05T00:04 | 15 min | 2026-11-05T00:19 | 4.5 min | 269 | 431 | 289078 | 12664 | complete |  |
| pilot_ragverify_hotpotqa | 2026-11-05T00:09 | 15 min | 2026-11-05T00:24 | 4.9 min | 294 | 443 | 283488 | 14318 | complete |  |
| pilot_ragverify_pubhealth | 2026-11-05T00:14 | 15 min | 2026-11-05T00:29 | 3.8 min | 226 | 400 | 285830 | 13621 | complete |  |
| pilot_crag_popqa_lt | 2026-11-05T00:27 | 10 min | 2026-11-05T00:37 | 3.0 min | 180 | 346 | 158026 | 1708 | complete |  |
| pilot_crag_triviaqa | 2026-11-05T00:41 | 10 min | 2026-11-05T00:51 | 3.3 min | 201 | 394 | 189560 | 2395 | complete |  |
| pilot_crag_hotpotqa | 2026-11-05T00:45 | 10 min | 2026-11-05T00:55 | 3.4 min | 204 | 400 | 186356 | 2922 | complete |  |
| pilot_crag_pubhealth | 2026-11-05T00:48 | 10 min | 2026-11-05T00:58 | 3.4 min | 202 | 400 | 184043 | 1844 | complete |  |
| pilot_selfrag_popqa_lt | 2026-11-05T00:59 | 10 min | 2026-11-05T01:09 | 1.2 min | 71 | 514 | 47750 | 10010 | complete |  |
| pilot_selfrag_triviaqa | 2026-11-05T01:00 | 10 min | 2026-11-05T01:10 | 2.0 min | 119 | 700 | 83959 | 16557 | complete |  |
| pilot_selfrag_hotpotqa | 2026-11-05T01:02 | 10 min | 2026-11-05T01:12 | 1.9 min | 113 | 700 | 90345 | 14692 | complete |  |
| pilot_selfrag_pubhealth | 2026-11-05T01:04 | 10 min | 2026-11-05T01:14 | 1.0 min | 59 | 700 | 105985 | 2738 | complete |  |
| pilot_rarr_popqa_lt | 2026-11-05T01:05 | 10 min | 2026-11-05T01:15 | 9.8 min | 589 | 775 | 834634 | 86656 | complete |  |
| pilot_rarr_triviaqa | 2026-11-05T01:15 | 10 min | 2026-11-05T01:25 | 10.5 min | 627 | 769 | 843463 | 94557 | complete |  |
| pilot_rarr_hotpotqa | 2026-11-05T01:25 | 10 min | 2026-11-05T01:35 | 10.4 min | 627 | 763 | 850166 | 96282 | complete |  |
| pilot_rarr_pubhealth | 2026-11-05T01:36 | 10 min | 2026-11-05T01:46 | 10.3 min | 616 | 752 | 817070 | 92855 | complete |  |
| pilot_ragverify_popqa_lt | 2026-11-05T01:46 | 15 min | 2026-11-05T02:01 | 0.0 min | 2 | 0 | 0 | 0 | complete |  |
| pilot_ragverify_triviaqa | 2026-11-05T01:46 | 15 min | 2026-11-05T02:01 | 0.0 min | 2 | 0 | 0 | 0 | complete |  |
| pilot_ragverify_hotpotqa | 2026-11-05T01:46 | 15 min | 2026-11-05T02:01 | 0.0 min | 2 | 0 | 0 | 0 | complete |  |
| pilot_ragverify_pubhealth | 2026-11-05T01:46 | 15 min | 2026-11-05T02:01 | 0.0 min | 1 | 0 | 0 | 0 | complete |  |
| devB_ragverify_popqa_lt | 2026-11-05T01:59 | 8 min | 2026-11-05T02:07 | 2.1 min | 124 | 493 | 167394 | 8024 | complete |  |
| devB_rag_topk_popqa_lt | 2026-11-05T02:02 | 8 min | 2026-11-05T02:10 | 0.8 min | 45 | 198 | 99818 | 957 | complete |  |
| devB_rag_rerank_popqa_lt | 2026-11-05T02:02 | 8 min | 2026-11-05T02:10 | 0.6 min | 33 | 133 | 68017 | 670 | complete |  |
| devB_rag_largek_popqa_lt | 2026-11-05T02:03 | 8 min | 2026-11-05T02:11 | 1.4 min | 84 | 198 | 190079 | 1018 | complete |  |
| devB_flare_popqa_lt | 2026-11-05T02:05 | 8 min | 2026-11-05T02:13 | 0.8 min | 46 | 78 | 39371 | 429 | complete |  |
| devB_crag_popqa_lt | 2026-11-05T02:07 | 8 min | 2026-11-05T02:15 | 4.0 min | 238 | 288 | 132107 | 1659 | complete |  |
| devB_selfrag_popqa_lt | 2026-11-05T02:11 | 8 min | 2026-11-05T02:19 | 3.6 min | 216 | 1386 | 128574 | 27082 | complete |  |
| devB_ragverify_triviaqa | 2026-11-05T02:15 | 8 min | 2026-11-05T02:23 | 2.5 min | 151 | 502 | 207180 | 10209 | complete |  |
| devB_rag_topk_triviaqa | 2026-11-05T02:17 | 8 min | 2026-11-05T02:25 | 1.2 min | 72 | 200 | 119337 | 979 | complete |  |
| devB_rag_rerank_triviaqa | 2026-11-05T02:19 | 8 min | 2026-11-05T02:27 | 0.8 min | 50 | 164 | 94959 | 730 | complete |  |
| devB_rag_largek_triviaqa | 2026-11-05T02:20 | 8 min | 2026-11-05T02:28 | 1.9 min | 113 | 200 | 219129 | 953 | complete |  |
| devB_flare_triviaqa | 2026-11-05T02:21 | 8 min | 2026-11-05T02:29 | 0.5 min | 30 | 55 | 30030 | 315 | complete |  |
| devB_crag_triviaqa | 2026-11-05T02:22 | 8 min | 2026-11-05T02:30 | 6.3 min | 380 | 400 | 195973 | 3166 | complete |  |
| devB_selfrag_triviaqa | 2026-11-05T02:28 | 8 min | 2026-11-05T02:36 | 4.4 min | 265 | 1400 | 165176 | 34550 | complete |  |
| devB_ragverify_hotpotqa | 2026-11-05T02:33 | 8 min | 2026-11-05T02:41 | 2.7 min | 161 | 436 | 163398 | 8247 | complete |  |
| devB_rag_topk_hotpotqa | 2026-11-05T02:36 | 8 min | 2026-11-05T02:44 | 0.9 min | 55 | 200 | 121869 | 991 | complete |  |
| devB_rag_rerank_hotpotqa | 2026-11-05T02:37 | 8 min | 2026-11-05T02:45 | 1.0 min | 59 | 171 | 101244 | 842 | complete |  |
| devB_rag_largek_hotpotqa | 2026-11-05T02:38 | 8 min | 2026-11-05T02:46 | 1.7 min | 102 | 200 | 229274 | 990 | complete |  |
| devB_flare_hotpotqa | 2026-11-05T02:39 | 8 min | 2026-11-05T02:47 | 0.5 min | 32 | 43 | 24825 | 274 | complete |  |
| devB_crag_hotpotqa | 2026-11-05T02:40 | 8 min | 2026-11-05T02:48 | 6.3 min | 377 | 400 | 196297 | 3658 | complete |  |
| devB_selfrag_hotpotqa | 2026-11-05T02:46 | 8 min | 2026-11-05T02:54 | 3.6 min | 218 | 1400 | 177471 | 28135 | complete |  |
| devB_ragverify_pubhealth | 2026-11-05T02:55 | 8 min | 2026-11-05T03:03 | 1.7 min | 102 | 329 | 99022 | 4983 | complete |  |
| devB_rag_topk_pubhealth | 2026-11-05T02:57 | 8 min | 2026-11-05T03:05 | 1.0 min | 61 | 200 | 126498 | 400 | complete |  |
| devB_rag_rerank_pubhealth | 2026-11-05T02:58 | 8 min | 2026-11-05T03:06 | 1.0 min | 57 | 187 | 117491 | 374 | complete |  |
| devB_rag_largek_pubhealth | 2026-11-05T02:59 | 8 min | 2026-11-05T03:07 | 1.9 min | 112 | 200 | 229999 | 400 | complete |  |
| devB_flare_pubhealth | 2026-11-05T03:01 | 8 min | 2026-11-05T03:09 | 0.2 min | 11 | 10 | 5565 | 20 | complete |  |
| devB_crag_pubhealth | 2026-11-05T03:01 | 8 min | 2026-11-05T03:09 | 6.7 min | 403 | 400 | 201886 | 3023 | complete |  |
| devB_selfrag_pubhealth | 2026-11-05T03:08 | 8 min | 2026-11-05T03:16 | 2.2 min | 132 | 1400 | 216091 | 5454 | complete |  |
| devB2_ragverify_popqa_lt | 2026-11-05T03:10 | 12 min | 2026-11-05T03:22 | 0.6 min | 35 | 52 | 30479 | 1511 | complete |  |
| devB2_ragverify_triviaqa | 2026-11-05T03:11 | 12 min | 2026-11-05T03:23 | 1.3 min | 78 | 158 | 103115 | 5023 | complete |  |
| devB2_ragverify_hotpotqa | 2026-11-05T03:12 | 12 min | 2026-11-05T03:24 | 1.4 min | 86 | 135 | 80427 | 4817 | complete |  |
| devB2_ragverify_pubhealth | 2026-11-05T03:14 | 12 min | 2026-11-05T03:26 | 1.4 min | 82 | 115 | 77384 | 4673 | complete |  |
| main_closed_book_popqa_lt | 2026-11-05T03:16 | 3 min | 2026-11-05T03:19 | 0.7 min | 41 | 988 | 52895 | 4650 | complete |  |
| main_rag_topk_popqa_lt | 2026-11-05T03:17 | 8 min | 2026-11-05T03:25 | 5.8 min | 350 | 991 | 638648 | 4481 | complete |  |
| main_rag_rerank_popqa_lt | 2026-11-05T03:23 | 16 min | 2026-11-05T03:39 | 6.4 min | 383 | 991 | 515920 | 4467 | complete |  |
| main_rag_largek_popqa_lt | 2026-11-05T03:30 | 12 min | 2026-11-05T03:42 | 14.4 min | 866 | 991 | 1417773 | 4319 | complete |  |
| main_rag_selfcheck_popqa_lt | 2026-11-05T03:44 | 10 min | 2026-11-05T03:54 | 9.8 min | 587 | 4955 | 3193240 | 23531 | complete |  |
| main_chain_of_note_popqa_lt | 2026-11-05T03:54 | 32 min | 2026-11-05T04:26 | 25.1 min | 1503 | 991 | 736757 | 157432 | complete |  |
| main_flare_popqa_lt | 2026-11-05T04:19 | 14 min | 2026-11-05T04:33 | 2.2 min | 132 | 338 | 161407 | 1667 | complete |  |
| main_cove_popqa_lt | 2026-11-05T04:21 | 34 min | 2026-11-05T04:55 | 22.8 min | 1366 | 4938 | 1175554 | 114255 | complete |  |
| main_rarr_popqa_lt | 2026-11-05T04:44 | 88 min | 2026-11-05T06:12 | 59.5 min | 3570 | 5223 | 6092561 | 388498 | complete |  |
| main_crag_popqa_lt | 2026-11-05T05:43 | 38 min | 2026-11-05T06:21 | 33.5 min | 2011 | 1979 | 927653 | 12403 | complete |  |
| main_ragverify_popqa_lt | 2026-11-05T06:17 | 50 min | 2026-11-05T07:07 | 11.4 min | 686 | 1188 | 666389 | 36368 | complete |  |
| main_selfrag_popqa_lt | 2026-11-05T06:28 | 22 min | 2026-11-05T06:50 | 20.8 min | 1251 | 6929 | 791337 | 125453 | complete |  |
| main_closed_book_triviaqa | 2026-11-05T06:49 | 3 min | 2026-11-05T06:52 | 1.0 min | 60 | 1000 | 63856 | 4466 | complete |  |
| main_rag_topk_triviaqa | 2026-11-05T06:51 | 8 min | 2026-11-05T06:59 | 8.5 min | 510 | 1000 | 829671 | 4572 | complete |  |
| main_rag_rerank_triviaqa | 2026-11-05T06:59 | 16 min | 2026-11-05T07:15 | 8.7 min | 521 | 1000 | 735708 | 4495 | complete |  |
| main_rag_largek_triviaqa | 2026-11-05T07:08 | 12 min | 2026-11-05T07:20 | 19.4 min | 1163 | 1000 | 1948404 | 4443 | complete |  |
| main_rag_selfcheck_triviaqa | 2026-11-05T07:27 | 10 min | 2026-11-05T07:37 | 12.0 min | 719 | 5000 | 4148355 | 22977 | complete |  |
| main_chain_of_note_triviaqa | 2026-11-05T07:39 | 32 min | 2026-11-05T08:11 | 28.6 min | 1719 | 1000 | 928671 | 173475 | complete |  |
| main_flare_triviaqa | 2026-11-05T08:08 | 14 min | 2026-11-05T08:22 | 2.0 min | 120 | 256 | 145140 | 1535 | complete |  |
| main_cove_triviaqa | 2026-11-05T08:10 | 34 min | 2026-11-05T08:44 | 25.6 min | 1538 | 4977 | 1399953 | 132160 | complete |  |
| main_rarr_triviaqa | 2026-11-05T08:36 | 88 min | 2026-11-05T10:04 | 76.3 min | 4580 | 5327 | 6444740 | 422974 | complete |  |
| main_crag_triviaqa | 2026-11-05T09:52 | 38 min | 2026-11-05T10:30 | 44.1 min | 2645 | 2000 | 994761 | 15752 | complete |  |
| main_ragverify_triviaqa | 2026-11-05T10:36 | 50 min | 2026-11-05T11:26 | 25.3 min | 1515 | 1533 | 1171957 | 53750 | complete |  |
| main_selfrag_triviaqa | 2026-11-05T11:01 | 22 min | 2026-11-05T11:23 | 70.7 min | 4240 | 6988 | 1083591 | 161892 | complete |  |
| main_closed_book_hotpotqa | 2026-11-05T12:12 | 3 min | 2026-11-05T12:15 | 1.9 min | 112 | 1000 | 64977 | 4575 | complete |  |
| main_rag_topk_hotpotqa | 2026-11-05T12:14 | 8 min | 2026-11-05T12:22 | 8.9 min | 533 | 1000 | 601329 | 4842 | complete |  |
| main_rag_rerank_hotpotqa | 2026-11-05T12:23 | 16 min | 2026-11-05T12:39 | 8.9 min | 531 | 999 | 584656 | 4634 | complete |  |
| main_rag_largek_hotpotqa | 2026-11-05T12:32 | 12 min | 2026-11-05T12:44 | 10.9 min | 653 | 1000 | 1124590 | 4803 | complete |  |
| main_rag_selfcheck_hotpotqa | 2026-11-05T12:43 | 10 min | 2026-11-05T12:53 | 16.7 min | 1000 | 5000 | 3006645 | 24225 | complete |  |
| main_chain_of_note_hotpotqa | 2026-11-05T13:25 | 32 min | 2026-11-05T13:57 | 6.3 min | 376 | 256 | 178898 | 43854 | complete |  |
| main_flare_hotpotqa | 2026-11-05T13:31 | 14 min | 2026-11-05T13:45 | 2.2 min | 132 | 289 | 170418 | 1484 | complete |  |
| main_cove_hotpotqa | 2026-11-05T13:33 | 34 min | 2026-11-05T14:07 | 80.0 min | 4800 | 4990 | 1418432 | 131334 | complete |  |
| main_rarr_hotpotqa | 2026-11-05T14:53 | 88 min | 2026-11-05T16:21 | 83.9 min | 5032 | 5374 | 6516462 | 435292 | complete |  |
| main_crag_hotpotqa | 2026-11-05T16:17 | 38 min | 2026-11-05T16:55 | 34.4 min | 2065 | 2000 | 985544 | 18039 | complete |  |
| main_ragverify_hotpotqa | 2026-11-05T16:52 | 50 min | 2026-11-05T17:42 | 22.1 min | 1328 | 1595 | 1010275 | 57566 | complete |  |
| main_selfrag_hotpotqa | 2026-11-05T17:14 | 22 min | 2026-11-05T17:36 | 24.3 min | 1460 | 7000 | 844953 | 159586 | complete |  |
| main_closed_book_pubhealth | 2026-11-05T17:38 | 3 min | 2026-11-05T17:41 | 0.7 min | 42 | 987 | 71375 | 1974 | complete |  |
| main_rag_topk_pubhealth | 2026-11-05T17:39 | 8 min | 2026-11-05T17:47 | 7.8 min | 470 | 987 | 802216 | 1974 | complete |  |
| main_rag_rerank_pubhealth | 2026-11-05T17:47 | 15 min | 2026-11-05T18:02 | 8.2 min | 489 | 986 | 759501 | 1972 | complete |  |
| main_rag_largek_pubhealth | 2026-11-05T17:55 | 12 min | 2026-11-05T18:07 | 14.7 min | 881 | 987 | 1523859 | 2020 | complete |  |
| main_rag_selfcheck_pubhealth | 2026-11-05T18:10 | 10 min | 2026-11-05T18:20 | 10.2 min | 613 | 4935 | 4011080 | 9870 | complete |  |
| main_chain_of_note_pubhealth | 2026-11-05T18:20 | 32 min | 2026-11-05T18:52 | 55.6 min | 3337 | 987 | 897955 | 213037 | complete |  |
| main_flare_pubhealth | 2026-11-05T19:16 | 13 min | 2026-11-05T19:29 | 0.8 min | 46 | 102 | 64711 | 204 | complete |  |
| main_cove_pubhealth | 2026-11-05T19:16 | 34 min | 2026-11-05T19:50 | 29.2 min | 1754 | 4924 | 1470604 | 152003 | complete |  |
| main_rarr_pubhealth | 2026-11-05T19:46 | 86 min | 2026-11-05T21:12 | 102.4 min | 6142 | 4969 | 6006213 | 396226 | complete |  |
| main_crag_pubhealth | 2026-11-05T21:55 | 37 min | 2026-11-05T22:32 | 40.8 min | 2447 | 1063 | 542079 | 8071 | complete |  |
| main_ragverify_pubhealth | 2026-11-05T22:36 | 49 min | 2026-11-05T23:25 | 34.8 min | 2086 | 1215 | 986502 | 50817 | complete |  |
| main_selfrag_pubhealth | 2026-11-05T23:11 | 22 min | 2026-11-05T23:33 | 17.4 min | 1045 | 6904 | 1268295 | 26228 | complete |  |
| main_closed_book_rgb | 2026-11-05T23:29 | 3 min | 2026-11-05T23:32 | 0.6 min | 34 | 403 | 22823 | 2535 | complete |  |
| main_rag_topk_rgb | 2026-11-05T23:29 | 11 min | 2026-11-05T23:40 | 27.7 min | 1660 | 1270 | 1713689 | 8047 | complete |  |
| main_rag_rerank_rgb | 2026-11-05T23:57 | 20 min | 2026-11-06T00:17 | 27.5 min | 1650 | 1241 | 1706711 | 7878 | complete |  |
| main_rag_largek_rgb | 2026-11-06T00:25 | 16 min | 2026-11-06T00:41 | 0.0 min | 1 | 0 | 0 | 0 | complete |  |
| main_rag_selfcheck_rgb | 2026-11-06T00:25 | 13 min | 2026-11-06T00:38 | 35.5 min | 2128 | 6350 | 8568445 | 40727 | complete |  |
| main_chain_of_note_rgb | 2026-11-06T01:00 | 42 min | 2026-11-06T01:42 | 496.3 min | 29780 | 1270 | 1839419 | 207471 | complete |  |
| main_flare_rgb | 2026-11-06T09:16 | 18 min | 2026-11-06T09:34 | 3.5 min | 212 | 223 | 212621 | 1659 | complete |  |
| main_cove_rgb | 2026-11-06T09:20 | 45 min | 2026-11-06T10:05 | 32.1 min | 1924 | 3580 | 1239439 | 87881 | complete |  |

> Note (2026-11-06 10:30): main_chain_of_note_rgb's actual time (496.3 min, wall-clock) includes the battery pause from 01:17:33 to 08:35:09 (437.6 min; logs/runner.log PAUSE/RESUME). No LLM requests were sent in that period. The GPU budget is therefore assessed on wall-clock minus logged battery/sleep pauses: 32.6 − 7.3 ≈ 25.3 GPU-h used as of 10:26. Heat-guard pauses stay counted. The table rows are left unchanged.

> Note (2026-11-06 11:55): the Claude desktop app updated and restarted at 10:37 (Application log: AppXDeployment + CoworkVMService stop/start). This killed the watchdog, the runner and llama-server, which were child processes of the app session. Idle 10:37–11:46. The first segment of main_rarr_rgb (09:52–10:37, ~45 min, items 1–1030) was never written to this ledger because the runner died before its end-of-job row. Add ≈0.75 GPU-h when totalling. The watchdog was relaunched at 11:46 via WMI (Win32_Process.Create), so it no longer depends on the app's process tree.
| main_rarr_rgb | 2026-11-06T11:46 | 114 min | 2026-11-06T13:40 | 19.2 min | 1155 | 1203 | 1497019 | 94667 | complete |  |
| main_crag_rgb | 2026-11-06T12:06 | 49 min | 2026-11-06T12:55 | 37.7 min | 2260 | 1644 | 1283734 | 12044 | complete |  |
| main_ragverify_rgb | 2026-11-06T12:44 | 65 min | 2026-11-06T13:49 | 59.2 min | 3553 | 1411 | 1899707 | 50995 | complete |  |
| main_selfrag_rgb | 2026-11-06T13:43 | 29 min | 2026-11-06T14:12 | 30.2 min | 1813 | 4051 | 1036826 | 97062 | complete |  |
| main_rag_selfcheck_s1_popqa_lt | 2026-11-06T14:13 | 10 min | 2026-11-06T14:23 | 14.3 min | 857 | 4955 | 3193240 | 22621 | complete |  |
| main_rag_selfcheck_s1_triviaqa | 2026-11-06T14:28 | 10 min | 2026-11-06T14:38 | 18.2 min | 1089 | 5000 | 4148355 | 22872 | complete |  |
| main_rag_selfcheck_s1_hotpotqa | 2026-11-06T14:46 | 10 min | 2026-11-06T14:56 | 13.6 min | 814 | 5000 | 3006645 | 24238 | complete |  |
| main_rag_selfcheck_s1_pubhealth | 2026-11-06T14:59 | 10 min | 2026-11-06T15:09 | 15.9 min | 957 | 4935 | 4011080 | 9870 | complete |  |
| main_rag_selfcheck_s1_rgb | 2026-11-06T15:43 | 13 min | 2026-11-06T15:56 | 33.7 min | 2021 | 5701 | 7643973 | 35843 | complete |  |
| main_rag_selfcheck_s2_popqa_lt | 2026-11-06T16:17 | 10 min | 2026-11-06T16:27 | 14.5 min | 872 | 4955 | 3193240 | 23715 | complete |  |
| main_rag_selfcheck_s2_triviaqa | 2026-11-06T16:31 | 10 min | 2026-11-06T16:41 | 18.6 min | 1117 | 5000 | 4148355 | 22952 | complete |  |
| main_rag_selfcheck_s2_hotpotqa | 2026-11-06T16:50 | 10 min | 2026-11-06T17:00 | 14.0 min | 838 | 5000 | 3006645 | 24088 | complete |  |
| main_rag_selfcheck_s2_pubhealth | 2026-11-06T17:46 | 10 min | 2026-11-06T17:56 | 12.4 min | 747 | 3716 | 3022313 | 7476 | complete |  |
| main_rag_selfcheck_s2_rgb | 2026-11-06T17:59 | 13 min | 2026-11-06T18:12 | 55.9 min | 3353 | 6350 | 8568445 | 40693 | complete |  |
| abl_ragverify_plusM1g09_popqa_lt | 2026-11-06T18:56 | 25 min | 2026-11-06T19:21 | 0.1 min | 4 | 0 | 0 | 0 | complete |  |
| abl_ragverify_noM2_popqa_lt | 2026-11-06T21:10 | 25 min | 2026-11-06T21:35 | 48.7 min | 2920 | 85 | 58571 | 2628 | complete |  |
| abl_ragverify_noM3_popqa_lt | 2026-11-06T21:59 | 25 min | 2026-11-06T22:24 | 5.2 min | 314 | 528 | 276380 | 2961 | complete |  |
| abl_ragverify_noM4_popqa_lt | 2026-11-06T22:04 | 25 min | 2026-11-06T22:29 | 0.0 min | 2 | 0 | 0 | 0 | complete |  |
| abl_ragverify_noM5_popqa_lt | 2026-11-06T22:04 | 25 min | 2026-11-06T22:29 | 0.0 min | 2 | 0 | 0 | 0 | complete |  |
| abl_ragverify_selfverify_popqa_lt | 2026-11-06T22:05 | 25 min | 2026-11-06T22:29 | 7.0 min | 418 | 864 | 494318 | 8194 | complete |  |
| abl_ragverify_hhem_popqa_lt | 2026-11-06T22:11 | 25 min | 2026-11-06T22:36 | 0.3 min | 20 | 0 | 0 | 0 | complete |  |
| abl_ragverify_k3_popqa_lt | 2026-11-06T22:12 | 25 min | 2026-11-06T22:37 | 6.1 min | 369 | 647 | 251878 | 16707 | complete |  |
| abl_ragverify_k10_popqa_lt | 2026-11-06T22:18 | 25 min | 2026-11-06T22:43 | 13.8 min | 827 | 669 | 651830 | 18411 | complete |  |
| abl_ragverify_rounds1_popqa_lt | 2026-11-06T22:32 | 25 min | 2026-11-06T22:57 | 0.0 min | 2 | 0 | 0 | 0 | complete |  |
| abl_ragverify_plusM1g09_hotpotqa | 2026-11-06T22:32 | 25 min | 2026-11-06T22:57 | 0.1 min | 3 | 0 | 0 | 0 | complete |  |
| abl_ragverify_noM2_hotpotqa | 2026-11-06T22:32 | 25 min | 2026-11-06T22:57 | 11.7 min | 705 | 763 | 501272 | 27870 | complete |  |
| abl_ragverify_noM3_hotpotqa | 2026-11-06T22:44 | 25 min | 2026-11-06T23:09 | 7.2 min | 433 | 661 | 391622 | 7157 | complete |  |
| abl_ragverify_noM4_hotpotqa | 2026-11-06T22:51 | 25 min | 2026-11-06T23:16 | 0.0 min | 2 | 0 | 0 | 0 | complete |  |
| abl_ragverify_noM5_hotpotqa | 2026-11-06T22:51 | 25 min | 2026-11-06T23:16 | 0.0 min | 2 | 0 | 0 | 0 | complete |  |
| abl_ragverify_selfverify_hotpotqa | 2026-11-06T22:51 | 25 min | 2026-11-06T23:16 | 9.9 min | 592 | 1235 | 840212 | 16194 | complete |  |
| abl_ragverify_hhem_hotpotqa | 2026-11-06T23:01 | 25 min | 2026-11-06T23:26 | 0.5 min | 29 | 8 | 5011 | 230 | complete |  |
| abl_ragverify_k3_hotpotqa | 2026-11-06T23:01 | 25 min | 2026-11-06T23:26 | 10.4 min | 626 | 953 | 456006 | 30564 | complete |  |
| abl_ragverify_k10_hotpotqa | 2026-11-06T23:12 | 25 min | 2026-11-06T23:37 | 15.3 min | 915 | 716 | 743670 | 24775 | complete |  |
| abl_ragverify_rounds1_hotpotqa | 2026-11-06T23:27 | 25 min | 2026-11-06T23:52 | 0.0 min | 3 | 0 | 0 | 0 | complete |  |
| sec3b_closed_book_popqa_lt | 2026-11-06T23:27 | 3 min | 2026-11-06T23:30 | 0.4 min | 22 | 499 | 37697 | 3514 | complete |  |
| sec3b_rag_topk_popqa_lt | 2026-11-06T23:28 | 4 min | 2026-11-06T23:32 | 1.9 min | 116 | 499 | 314359 | 2531 | complete |  |
| sec3b_rag_rerank_popqa_lt | 2026-11-06T23:30 | 8 min | 2026-11-06T23:38 | 2.1 min | 125 | 499 | 258085 | 2729 | complete |  |
| sec3b_flare_popqa_lt | 2026-11-06T23:32 | 7 min | 2026-11-06T23:39 | 2.7 min | 164 | 537 | 244662 | 3270 | complete |  |
| sec3b_cove_popqa_lt | 2026-11-06T23:35 | 17 min | 2026-11-06T23:52 | 8.0 min | 480 | 2371 | 580842 | 56134 | complete |  |
| sec3b_rag_selfcheck_popqa_lt | 2026-11-06T23:43 | 5 min | 2026-11-06T23:48 | 3.5 min | 209 | 2495 | 1571795 | 16528 | complete |  |
| sec3b_crag_popqa_lt | 2026-11-06T23:46 | 19 min | 2026-11-07T00:05 | 7.3 min | 439 | 998 | 484855 | 18196 | complete |  |
| sec3b_ragverify_popqa_lt | 2026-11-06T23:53 | 25 min | 2026-11-07T00:18 | 4.0 min | 239 | 609 | 336304 | 18190 | complete |  |
| sec3b_closed_book_hotpotqa | 2026-11-06T23:57 | 3 min | 2026-11-07T00:00 | 0.4 min | 23 | 500 | 43258 | 3307 | complete |  |
| sec3b_rag_topk_hotpotqa | 2026-11-06T23:58 | 4 min | 2026-11-07T00:02 | 2.2 min | 134 | 500 | 300519 | 2458 | complete |  |
| sec3b_rag_rerank_hotpotqa | 2026-11-07T00:00 | 8 min | 2026-11-07T00:08 | 2.2 min | 131 | 499 | 288638 | 2436 | complete |  |
| sec3b_flare_hotpotqa | 2026-11-07T00:02 | 7 min | 2026-11-07T00:09 | 3.2 min | 190 | 515 | 305357 | 2663 | complete |  |
| sec3b_cove_hotpotqa | 2026-11-07T00:05 | 17 min | 2026-11-07T00:22 | 9.7 min | 582 | 2425 | 705894 | 65492 | complete |  |
| sec3b_rag_selfcheck_hotpotqa | 2026-11-07T00:15 | 5 min | 2026-11-07T00:20 | 3.3 min | 197 | 2500 | 1502595 | 14953 | complete |  |
| sec3b_crag_hotpotqa | 2026-11-07T00:19 | 19 min | 2026-11-07T00:38 | 6.6 min | 397 | 1000 | 462999 | 18609 | complete |  |
| sec3b_ragverify_hotpotqa | 2026-11-07T00:25 | 25 min | 2026-11-07T00:50 | 4.9 min | 294 | 794 | 495295 | 26493 | complete |  |
| sec3b_closed_book_pubhealth | 2026-11-07T00:30 | 3 min | 2026-11-07T00:33 | 0.2 min | 10 | 500 | 46800 | 1027 | complete |  |
| sec3b_rag_topk_pubhealth | 2026-11-07T00:30 | 4 min | 2026-11-07T00:34 | 2.8 min | 169 | 500 | 402374 | 1188 | complete |  |
| sec3b_rag_rerank_pubhealth | 2026-11-07T00:33 | 8 min | 2026-11-07T00:41 | 2.7 min | 165 | 499 | 382295 | 1193 | complete |  |
| sec3b_flare_pubhealth | 2026-11-07T00:36 | 7 min | 2026-11-07T00:43 | 3.1 min | 184 | 656 | 404499 | 1564 | complete |  |
| sec3b_cove_pubhealth | 2026-11-07T00:39 | 17 min | 2026-11-07T00:56 | 10.8 min | 651 | 2361 | 714835 | 67495 | complete |  |
| sec3b_rag_selfcheck_pubhealth | 2026-11-07T00:50 | 5 min | 2026-11-07T00:55 | 3.3 min | 199 | 2500 | 2011870 | 6417 | complete |  |
| sec3b_crag_pubhealth | 2026-11-07T00:53 | 19 min | 2026-11-07T01:12 | 7.8 min | 468 | 1000 | 468958 | 8251 | complete |  |
| sec3b_ragverify_pubhealth | 2026-11-07T01:01 | 25 min | 2026-11-07T01:26 | 5.8 min | 346 | 614 | 494013 | 27559 | complete |  |
| quant3bq8_closed_book_popqa_lt | 2026-11-07T01:07 | 3 min | 2026-11-07T01:10 | 0.2 min | 9 | 200 | 15121 | 1357 | complete |  |
| quant3bq8_rag_topk_popqa_lt | 2026-11-07T01:07 | 3 min | 2026-11-07T01:10 | 0.9 min | 52 | 200 | 126548 | 1235 | complete |  |
| quant3bq8_ragverify_popqa_lt | 2026-11-07T01:08 | 10 min | 2026-11-07T01:18 | 1.1 min | 68 | 279 | 151599 | 7652 | complete |  |
| quant3bq8_closed_book_hotpotqa | 2026-11-07T01:09 | 3 min | 2026-11-07T01:12 | 0.2 min | 10 | 200 | 17207 | 1453 | complete |  |
| quant3bq8_rag_topk_hotpotqa | 2026-11-07T01:09 | 3 min | 2026-11-07T01:12 | 1.1 min | 69 | 200 | 119215 | 1174 | complete |  |
| quant3bq8_ragverify_hotpotqa | 2026-11-07T01:10 | 10 min | 2026-11-07T01:20 | 1.7 min | 99 | 367 | 230136 | 10937 | complete |  |
| abl_ragverify_minicheckdeb_popqa_lt | 2026-11-07T01:13 | 25 min | 2026-11-07T01:38 | 1.0 min | 62 | 19 | 10451 | 364 | complete |  |
| abl_ragverify_minicheckdeb_hotpotqa | 2026-11-07T01:15 | 25 min | 2026-11-07T01:40 | 1.8 min | 109 | 77 | 52608 | 2279 | complete |  |
| rob_rag_topk_hotpotqa_dist00 | 2026-11-07T01:18 | 3 min | 2026-11-07T01:21 | 0.0 min | 0 | 0 | 0 | 0 | complete |  |
| rob_rag_rerank_hotpotqa_dist00 | 2026-11-07T01:18 | 5 min | 2026-11-07T01:23 | 2.9 min | 172 | 279 | 169973 | 1344 | complete |  |
| rob_chain_of_note_hotpotqa_dist00 | 2026-11-07T01:21 | 10 min | 2026-11-07T01:31 | 0.0 min | 1 | 0 | 0 | 0 | complete |  |
| rob_crag_hotpotqa_dist00 | 2026-11-07T01:21 | 11 min | 2026-11-07T01:32 | 4.5 min | 267 | 290 | 239633 | 1490 | complete |  |
| rob_selfrag_hotpotqa_dist00 | 2026-11-07T01:26 | 7 min | 2026-11-07T01:33 | 0.0 min | 2 | 0 | 0 | 0 | complete |  |
| rob_ragverify_hotpotqa_dist00 | 2026-11-07T01:26 | 15 min | 2026-11-07T01:41 | 5.2 min | 313 | 390 | 229274 | 14284 | complete |  |
| rob_rag_topk_hotpotqa_dist25 | 2026-11-07T01:31 | 3 min | 2026-11-07T01:34 | 3.3 min | 199 | 288 | 186507 | 1458 | complete |  |
| rob_rag_rerank_hotpotqa_dist25 | 2026-11-07T01:34 | 5 min | 2026-11-07T01:39 | 3.1 min | 185 | 236 | 151504 | 1250 | complete |  |
| rob_chain_of_note_hotpotqa_dist25 | 2026-11-07T01:38 | 10 min | 2026-11-07T01:48 | 18.6 min | 1119 | 288 | 215019 | 49854 | complete |  |
| rob_crag_hotpotqa_dist25 | 2026-11-07T01:56 | 11 min | 2026-11-07T02:07 | 6.1 min | 367 | 277 | 233434 | 1457 | complete |  |
| rob_selfrag_hotpotqa_dist25 | 2026-11-07T02:02 | 7 min | 2026-11-07T02:09 | 1.6 min | 98 | 268 | 52642 | 6317 | complete |  |
| rob_ragverify_hotpotqa_dist25 | 2026-11-07T02:04 | 15 min | 2026-11-07T02:19 | 6.4 min | 383 | 412 | 255657 | 15044 | complete |  |
| rob_rag_topk_hotpotqa_dist50 | 2026-11-07T02:11 | 3 min | 2026-11-07T02:14 | 3.8 min | 230 | 300 | 214735 | 1418 | complete |  |
| rob_rag_rerank_hotpotqa_dist50 | 2026-11-07T02:15 | 5 min | 2026-11-07T02:20 | 3.7 min | 221 | 269 | 192732 | 1323 | complete |  |
| rob_chain_of_note_hotpotqa_dist50 | 2026-11-07T02:18 | 10 min | 2026-11-07T02:28 | 12.4 min | 741 | 300 | 244435 | 52287 | complete |  |
| rob_crag_hotpotqa_dist50 | 2026-11-07T02:31 | 11 min | 2026-11-07T02:42 | 7.1 min | 427 | 298 | 260119 | 1560 | complete |  |
| rob_selfrag_hotpotqa_dist50 | 2026-11-07T02:38 | 7 min | 2026-11-07T02:45 | 3.7 min | 221 | 543 | 107176 | 13335 | complete |  |
| rob_ragverify_hotpotqa_dist50 | 2026-11-07T02:42 | 15 min | 2026-11-07T02:57 | 7.1 min | 429 | 468 | 304468 | 16521 | complete |  |
| rob_rag_topk_hotpotqa_dist75 | 2026-11-07T02:49 | 3 min | 2026-11-07T02:52 | 4.0 min | 237 | 300 | 226887 | 1603 | complete |  |
| rob_rag_rerank_hotpotqa_dist75 | 2026-11-07T02:53 | 5 min | 2026-11-07T02:58 | 3.8 min | 228 | 277 | 209142 | 1370 | complete |  |
| rob_chain_of_note_hotpotqa_dist75 | 2026-11-07T02:57 | 10 min | 2026-11-07T03:07 | 12.9 min | 772 | 300 | 256587 | 53703 | complete |  |
| rob_crag_hotpotqa_dist75 | 2026-11-07T03:10 | 11 min | 2026-11-07T03:21 | 6.3 min | 380 | 279 | 246009 | 1484 | complete |  |
| rob_selfrag_hotpotqa_dist75 | 2026-11-07T03:16 | 7 min | 2026-11-07T03:23 | 1.7 min | 102 | 267 | 55104 | 6571 | complete |  |
| rob_ragverify_hotpotqa_dist75 | 2026-11-07T03:18 | 15 min | 2026-11-07T03:33 | 8.1 min | 485 | 547 | 352220 | 18947 | complete |  |
