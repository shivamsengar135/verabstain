"""Probe Hugging Face repos: existence, gating, licence tag, and sizes of matching files.

Usage: python -I scripts/setup/probe_hf.py > logs/probe_hf.txt
Writes one compact line per repo plus matching files. Stdlib only.
"""
import json
import re
import urllib.request

REPOS = [
    ("model", "Qwen/Qwen2.5-7B-Instruct-GGUF", r"q4_k_m"),
    ("model", "Qwen/Qwen2.5-7B-Instruct", r"LICENSE|config.json"),
    ("model", "bartowski/Llama-3.2-3B-Instruct-GGUF", r"Q4_K_M|Q8_0|f16"),
    ("model", "meta-llama/Llama-3.2-3B-Instruct", r"config.json"),
    ("model", "selfrag/selfrag_llama2_7b", r"\.(bin|safetensors|json|model)$"),
    ("model", "lytang/MiniCheck-Flan-T5-Large", r"\.(bin|safetensors)$"),
    ("model", "lytang/MiniCheck-RoBERTa-Large", r"\.(bin|safetensors)$"),
    ("model", "lytang/MiniCheck-DeBERTa-v3-Large", r"\.(bin|safetensors)$"),
    ("model", "bespokelabs/Bespoke-MiniCheck-7B", r"\.safetensors$"),
    ("model", "yzha/AlignScore", r"\.ckpt$"),
    ("model", "vectara/hallucination_evaluation_model", r"\.(bin|safetensors|py)$"),
    ("model", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", r"\.(bin|safetensors)$"),
    ("model", "intfloat/e5-small-v2", r"\.(bin|safetensors)$"),
    ("model", "BAAI/bge-small-en-v1.5", r"\.(bin|safetensors)$"),
    ("model", "BAAI/bge-reranker-base", r"\.(bin|safetensors)$"),
    ("model", "cross-encoder/ms-marco-MiniLM-L6-v2", r"\.(bin|safetensors)$"),
    ("dataset", "BeIR/hotpotqa", r"corpus|queries"),
    ("dataset", "BeIR/fever", r"corpus"),
    ("dataset", "hotpotqa/hotpot_qa", r"distractor|fullwiki"),
    ("dataset", "mandarjoshi/trivia_qa", r"unfiltered.nocontext|rc.nocontext"),
    ("dataset", "akariasai/PopQA", r"."),
    ("dataset", "lytang/LLM-AggreFact", r"."),
    ("dataset", "Salesforce/FaithEval-unanswerable-v1.0", r"."),
    ("dataset", "Salesforce/FaithEval-counterfactual-v1.0", r"."),
    ("dataset", "Salesforce/FaithEval-inconsistent-v1.0", r"."),
    ("dataset", "ImperialCollegeLondon/health_fact", r"."),
    ("dataset", "fever/fever", r"."),
    ("dataset", "selfrag/selfrag_train_data", r"."),
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ragverify-setup"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


for kind, repo, pat in REPOS:
    base = "models" if kind == "model" else "datasets"
    try:
        info = get(f"https://huggingface.co/api/{base}/{repo}?blobs=true")
    except Exception as e:  # noqa: BLE001
        print(f"[{kind}] {repo}: ERROR {e}")
        continue
    card = info.get("cardData") or {}
    lic = card.get("license") or [t for t in info.get("tags", []) if t.startswith("license:")]
    sib = [(s["rfilename"], s.get("size") or 0) for s in info.get("siblings", [])]
    hits = [(n, sz) for n, sz in sib if re.search(pat, n, re.I)]
    sha = (info.get("sha") or "")[:12]
    print(f"[{kind}] {repo}: gated={info.get('gated')} license={lic} sha={sha} files={len(sib)}")
    for n, sz in hits[:12]:
        print(f"      {n}  {sz/1e9:.2f} GB")
