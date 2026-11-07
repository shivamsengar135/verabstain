"""Download pinned models and datasets into models/ and data/raw/, logging provenance.

Usage: .venv/Scripts/python.exe scripts/setup/download_assets.py GROUP [GROUP ...]
Groups: core, data, verifiers, secondary, selfrag, crag
Each asset goes into its own directory. For every file the manifest
logs/downloads_manifest.tsv records: time, source, revision, local path, bytes, sha256.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

MANIFEST = ROOT / "logs" / "downloads_manifest.tsv"

HF = {
    "core": [
        ("model", "Qwen/Qwen2.5-7B-Instruct-GGUF", ["qwen2.5-7b-instruct-q4_k_m-0000*-of-00002.gguf", "LICENSE", "README.md"], "models/qwen2.5-7b-it-gguf"),
        ("dataset", "BeIR/hotpotqa", ["corpus/*", "queries/*", "README.md"], "data/raw/beir_hotpotqa"),
        ("model", "BAAI/bge-small-en-v1.5", ["*.json", "*.txt", "model.safetensors", "1_Pooling/*", "README.md"], "models/bge-small-en-v1.5"),
        ("model", "cross-encoder/ms-marco-MiniLM-L6-v2", ["*.json", "*.txt", "model.safetensors", "README.md"], "models/msmarco-minilm-l6-v2"),
    ],
    "data": [
        ("dataset", "hotpotqa/hotpot_qa", ["distractor/validation-*", "distractor/train-00000-*", "README.md"], "data/raw/hotpotqa"),
        ("dataset", "mandarjoshi/trivia_qa", ["unfiltered.nocontext/validation-*", "unfiltered.nocontext/train-*", "README.md"], "data/raw/triviaqa"),
        ("dataset", "akariasai/PopQA", ["test.tsv", "README.md"], "data/raw/popqa"),
        ("dataset", "Salesforce/FaithEval-unanswerable-v1.0", ["data/*", "README.md"], "data/raw/faitheval_unans"),
        ("dataset", "Salesforce/FaithEval-counterfactual-v1.0", ["data/*", "README.md"], "data/raw/faitheval_cf"),
        ("dataset", "Salesforce/FaithEval-inconsistent-v1.0", ["data/*", "README.md"], "data/raw/faitheval_inc"),
    ],
    "verifiers": [
        ("model", "lytang/MiniCheck-Flan-T5-Large", ["*.json", "*.model", "pytorch_model.bin", "README.md"], "models/minicheck-ft5-l"),
        ("model", "lytang/MiniCheck-DeBERTa-v3-Large", ["*.json", "*.model", "pytorch_model.bin", "README.md"], "models/minicheck-deb-l"),
        ("model", "vectara/hallucination_evaluation_model", ["*.json", "*.py", "model.safetensors", "README.md"], "models/hhem-2.1-open"),
        ("model", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", ["*.json", "*.model", "model.safetensors", "README.md"], "models/deberta-v3-l-nli"),
        ("model", "google/flan-t5-base", ["*.json", "*.model", "README.md"], "models/flan-t5-base-tok"),
    ],
    "verifiers_late": [
        ("model", "FacebookAI/roberta-large", ["*.json", "*.txt", "README.md"], "models/roberta-large-tok"),
        ("model", "yzha/AlignScore", ["AlignScore-large.ckpt", "README.md"], "models/alignscore"),
        ("model", "lytang/MiniCheck-RoBERTa-Large", ["*.json", "*.txt", "pytorch_model.bin", "README.md"], "models/minicheck-rbt-l"),
    ],
    "secondary": [
        ("model", "bartowski/Llama-3.2-3B-Instruct-GGUF", ["Llama-3.2-3B-Instruct-Q4_K_M.gguf", "README.md"], "models/llama3.2-3b-it-gguf"),
    ],
    "selfrag": [
        ("model", "selfrag/selfrag_llama2_7b", ["*.json", "*.model", "README.md"], "models/selfrag-7b-hf"),
        ("model", "QuantFactory/selfrag_llama2_7b-GGUF", ["selfrag_llama2_7b.Q4_K_M.gguf", "README.md"], "models/selfrag-7b-gguf"),
    ],
}

URLS = {
    "data": [
        ("https://raw.githubusercontent.com/chen700564/RGB/master/data/en.json", "data/raw/rgb/en.json"),
        ("https://raw.githubusercontent.com/chen700564/RGB/master/data/en_int.json", "data/raw/rgb/en_int.json"),
        ("https://raw.githubusercontent.com/chen700564/RGB/master/data/en_fact.json", "data/raw/rgb/en_fact.json"),
        ("https://raw.githubusercontent.com/chen700564/RGB/master/README.md", "data/raw/rgb/README.md"),
        ("https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl", "data/raw/ragtruth/response.jsonl"),
        ("https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl", "data/raw/ragtruth/source_info.jsonl"),
        ("https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/README.md", "data/raw/ragtruth/README.md"),
    ],
}

GDRIVE = {
    "data": [("file", "1TLKhWjez63H4uBtgCxyoyJsZi-IMgnDb", "data/raw/selfrag_eval/eval_data.zip")],
    "crag": [("folder", "1CRFGsyNguXJwKSvFvJm_82GOOlkWSkW7", "models/crag-evaluator")],
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def log(source: str, rev: str, path: Path) -> None:
    MANIFEST.parent.mkdir(exist_ok=True)
    new = not MANIFEST.exists()
    with MANIFEST.open("a", encoding="utf-8") as m:
        if new:
            m.write("time\tsource\trevision\tpath\tbytes\tsha256\n")
        now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        m.write(f"{now}\t{source}\t{rev}\t{path.relative_to(ROOT).as_posix()}\t{path.stat().st_size}\t{sha256(path)}\n")


def fetch_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    req = urllib.request.Request(url, headers={"User-Agent": "ragverify-setup"})
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=600) as r, tmp.open("wb") as f:
        while b := r.read(1 << 20):
            f.write(b)
    tmp.rename(dest)
    log(url, "-", dest)


def main(groups: list[str]) -> None:
    api = HfApi()
    for g in groups:
        for kind, repo, patterns, local in HF.get(g, []):
            info = api.repo_info(repo, repo_type=kind)
            dest = ROOT / local
            t0 = dt.datetime.now()
            snapshot_download(repo_id=repo, repo_type=kind, revision=info.sha, allow_patterns=patterns, local_dir=dest)
            for p in sorted(dest.rglob("*")):
                if p.is_file() and ".cache" not in p.parts:
                    log(f"hf:{kind}:{repo}", info.sha, p)
            secs = (dt.datetime.now() - t0).total_seconds()
            print(f"[{g}] {repo} -> {local} ({secs:.0f}s)", flush=True)
        for url, local in URLS.get(g, []):
            try:
                fetch_url(url, ROOT / local)
                print(f"[{g}] {url} -> {local}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{g}] FAILED {url}: {e}", flush=True)
        for kind, gid, local in GDRIVE.get(g, []):
            import gdown
            dest = ROOT / local
            try:
                if kind == "file":
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        gdown.download(id=gid, output=str(dest), quiet=True)
                    log(f"gdrive:file:{gid}", "-", dest)
                else:
                    dest.mkdir(parents=True, exist_ok=True)
                    gdown.download_folder(id=gid, output=str(dest), quiet=True)
                    for p in sorted(dest.rglob("*")):
                        if p.is_file():
                            log(f"gdrive:folder:{gid}", "-", p)
                print(f"[{g}] gdrive {gid} -> {local}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{g}] FAILED gdrive {gid}: {e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
