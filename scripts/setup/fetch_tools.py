"""Download pinned tool binaries (uv, llama.cpp CUDA build) into tools/ and log SHA-256.

Usage: python -I scripts/setup/fetch_tools.py
Every file goes into its own fresh directory under tools/. Hashes are appended to
tools/MANIFEST.tsv and copied into ENVIRONMENT.md by hand.
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
LLAMA_BUILD = "b12110"
ITEMS = [
    ("uv-0.13.7", "https://github.com/astral-sh/uv/releases/download/0.13.7/uv-x86_64-pc-windows-msvc.zip"),
    (f"llama-{LLAMA_BUILD}", f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-win-cuda-12.4-x64.zip"),
    (f"llama-{LLAMA_BUILD}", f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/cudart-llama-bin-win-cuda-12.4-x64.zip"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    TOOLS.mkdir(exist_ok=True)
    manifest = TOOLS / "MANIFEST.tsv"
    for name, url in ITEMS:
        dest_dir = TOOLS / name
        dest_dir.mkdir(exist_ok=True)
        zpath = TOOLS / "_zips" / url.rsplit("/", 1)[1]
        zpath.parent.mkdir(exist_ok=True)
        if not zpath.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "ragverify-setup"})
            with urllib.request.urlopen(req, timeout=300) as r, zpath.with_suffix(".part").open("wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            zpath.with_suffix(".part").rename(zpath)
        digest = sha256(zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest_dir)
        with manifest.open("a", encoding="utf-8") as m:
            m.write(f"{name}\t{url}\t{digest}\t{zpath.stat().st_size}\n")
        print(f"{name}: {zpath.name} {zpath.stat().st_size/1e6:.1f} MB sha256={digest[:16]}...")


if __name__ == "__main__":
    sys.exit(main())
