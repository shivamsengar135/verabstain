#!/usr/bin/env bash
# Recreate the project venv from requirements.lock. Run from the project root in Git Bash.
set -euo pipefail
ROOT="$(pwd)"
export UV_CACHE_DIR="$ROOT/.cache/uv" PYTHONUTF8=1
UV="$ROOT/tools/uv-0.13.7/uv.exe"
PY311="/c/Users/shiva/AppData/Local/Programs/Python/Python311/python.exe"
[ -d .venv ] || "$UV" venv .venv --python "$PY311"
"$UV" pip compile requirements.in -o requirements.lock --python .venv/Scripts/python.exe --torch-backend cu130 --python-platform x86_64-pc-windows-msvc
"$UV" pip sync requirements.lock --python .venv/Scripts/python.exe --torch-backend cu130
.venv/Scripts/python.exe -c "import torch,sys;print(sys.version.split()[0],torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
