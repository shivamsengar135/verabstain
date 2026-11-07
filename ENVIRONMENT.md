# ENVIRONMENT — RAG-Verify

Measured 2026-11-04 (T0 = 21:46 IST). All tools, caches and models live inside the project folder.

## Hardware and OS (measured)

| Item | Value | Source |
|---|---|---|
| Machine | HP OMEN by HP 16.1 inch Gaming Laptop PC | `Win32_ComputerSystem` |
| CPU | AMD Ryzen 7 7840HS w/ Radeon 780M Graphics, 8 cores / 16 threads | `Win32_Processor` |
| RAM | 15.29 GB visible; only 3–5.5 GB free with the user's other apps open | `Win32_OperatingSystem` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, driver 610.62 (CUDA 13.x capable); dGPU idle at start (0 MiB used → display on iGPU) | `nvidia-smi` |
| OS | Windows 11 Home Single Language, 10.0.26200 | `Win32_OperatingSystem` |
| Disk | C: 570 GB free at T0 | `Get-PSDrive` |
| Power | on AC at T0 (battery 74%) | `Win32_Battery` |
| Long paths | disabled (`LongPathsEnabled=0`) → short paths, `local_dir` downloads | registry (read-only) |
| WSL | not installed (`wsl -l -v` → `REGDB_E_CLASSNOTREG`) | `wsl.exe` |
| Network | ≈ 3 MB/s sustained (GitHub and Hugging Face) | measured during downloads |

## Software

- Python 3.11.9 (user install `C:\Users\shiva\AppData\Local\Programs\Python\Python311`), venv `.venv` created by **uv 0.13.7** (standalone binary in `tools/uv-0.13.7/`).
- Lockfile: `requirements.lock` (uv pip compile from `requirements.in`, torch from the cu130 index). Recreate with `bash scripts/setup/make_venv.sh`.
- Key versions: torch 2.14.1+cu130 (`torch.cuda.is_available()` = True on the RTX 4060), transformers 5.22.1, sentence-transformers 6.2.0, faiss-cpu 1.15.1, numpy 2.4.6, scipy 1.17.1, pandas 3.1.1, pyarrow 26.0.1, scikit-learn 1.9.1, huggingface_hub 2.4.0, PyStemmer 3.1.0.
- Inference backend: **llama.cpp build b12110** (commit cd959bae9, official Windows CUDA 12.4 x64 release + cudart), `tools/llama-b12110/llama-server.exe`. SHA-256 of release zips in `tools/MANIFEST.tsv`.
- Downloads and their SHA-256: `logs/downloads_manifest.tsv` (every model and data file, with HF commit).

## Backend choice (tested, not assumed)

- vLLM: no official native Windows support (only unofficial community builds); WSL2 is not installed → not used unless the user installs WSL.
- llama.cpp `llama-server`: official Windows CUDA build; OpenAI-compatible chat endpoint with `logprobs`/`top_logprobs`, native `/completion` with `n_probs`, parallel slots with continuous batching, unified KV cache with q8_0 quantization. **Chosen** — see benchmark below.
- Hugging Face transformers: fallback only.

## Throughput benchmark (2026-11-04 23:10, `scripts/bench_backend.py`, raw: `logs/bench_qwen7b_q4_c32768_p8.json`)

Qwen2.5-7B-Instruct Q4_K_M, llama-server b12110, `-c 32768 -np 8 -fa on -ctk q8_0 -ctv q8_0 --kv-unified -b 2048 -ub 512`.
RAG-length prompt = 5 passages × 120 words + question ≈ 739 prompt tokens, ≈ 67 generated tokens, prompt cache off,
FAISS index build running on the CPU at the same time.

| Concurrent requests | Prompt processing tok/s (per request, median) | Generation tok/s (per request) | Aggregate generation tok/s | Requests/s |
|---|---|---|---|---|
| 1 | 2,583 | 42.9 | 35.9 | 0.53 |
| 4 | 2,326 | 23.9 | 76.8 | 1.13 |
| 8 | 1,578 | 12.5 | 86.9 | 1.30 |

VRAM with the model loaded: 5,403 MiB (peak 5,423 MiB) of 8,188 MiB, which leaves room for a ~1.8 GB verifier on the same GPU (MiniCheck-Flan-T5-Large bf16 peaked at 1,815 MiB with batch 16 × ~600-word inputs). Chat `logprobs`/`top_logprobs` and native `/completion` `n_probs` both return token log-probabilities. Settings kept: 8 slots, 32k unified q8_0 KV.

## How to run

```bash
export PYTHONUTF8=1
.venv/Scripts/python.exe scripts/llm_server.py start qwen7b_q4      # detached llama-server on 127.0.0.1:8091
.venv/Scripts/python.exe scripts/run_queue.py --queue jobs/queue.yaml
.venv/Scripts/python.exe scripts/status.py
```

Environment variables used by the scripts: `PYTHONUTF8=1`, `HF_HOME=<project>/.cache/hf`, `HF_HUB_DISABLE_XET=1` (the Xet transfer backend stalled on this network).
