"""Start/stop/status for a detached llama-server process (one LLM resident at a time).

Usage:
  .venv/Scripts/python.exe scripts/llm_server.py start PROFILE [--port 8091] [--ctx N] [--parallel N]
  .venv/Scripts/python.exe scripts/llm_server.py stop
  .venv/Scripts/python.exe scripts/llm_server.py status
State is kept in logs/server/current.json; server output goes to logs/server/<profile>-<time>.log.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "logs" / "server" / "current.json"
EXE = ROOT / "tools" / "llama-b12110" / "llama-server.exe"


def profiles() -> dict:
    return yaml.safe_load((ROOT / "jobs" / "servers.yaml").read_text(encoding="utf-8"))


def current() -> dict | None:
    if not STATE.exists():
        return None
    st = json.loads(STATE.read_text(encoding="utf-8"))
    if psutil.pid_exists(st["pid"]):
        try:
            if "llama-server" in psutil.Process(st["pid"]).name():
                return st
        except psutil.Error:
            pass
    STATE.unlink(missing_ok=True)
    return None


def healthy(port: int) -> bool:
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


def stop() -> None:
    st = current()
    if not st:
        return
    try:
        p = psutil.Process(st["pid"])
        p.terminate()
        p.wait(timeout=30)
    except psutil.TimeoutExpired:
        psutil.Process(st["pid"]).kill()
    except psutil.Error:
        pass
    STATE.unlink(missing_ok=True)
    time.sleep(3)  # let the driver release VRAM


CPU_SET = [0, 1, 2, 3]


def start(name: str, port: int = 8091, ctx: int | None = None, parallel: int | None = None,
          timeout: float = 600) -> dict:
    prof = profiles()[name]
    st = current()
    if st and st["profile"] == name and st.get("ctx") == (ctx or prof["ctx"]) and \
            st.get("parallel") == (parallel or prof["parallel"]) and healthy(st["port"]):
        return st
    stop()
    ctx, parallel = ctx or prof["ctx"], parallel or prof["parallel"]
    cmd = [str(EXE), "-m", str(ROOT / prof["model"]), "--host", "127.0.0.1", "--port", str(port),
           "-c", str(ctx), "-np", str(parallel), *prof["args"]]
    log = ROOT / "logs" / "server" / f"{name}-{datetime.now():%Y%m%d-%H%M%S}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             creationflags=flags, close_fds=True, cwd=str(ROOT))
    try:  # heat control (2026-11-05 22:00): keep the server on 4 logical cores so the CPU boosts less; timing only
        import psutil
        psutil.Process(p.pid).cpu_affinity(CPU_SET)
    except Exception:  # noqa: BLE001
        pass
    st = {"pid": p.pid, "profile": name, "tag": prof["tag"], "port": port, "ctx": ctx, "parallel": parallel,
          "started": datetime.now().astimezone().isoformat(timespec="seconds"), "log": str(log.relative_to(ROOT)),
          "cmd": cmd}
    STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if p.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {p.returncode}); see {log}")
        if healthy(port):
            st["load_seconds"] = round(time.time() - t0, 1)
            STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")
            return st
        time.sleep(2)
    stop()
    raise TimeoutError(f"llama-server not healthy after {timeout}s; see {log}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop", "status"])
    ap.add_argument("profile", nargs="?")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--parallel", type=int)
    a = ap.parse_args()
    if a.action == "start":
        print(json.dumps(start(a.profile, a.port, a.ctx, a.parallel), indent=1))
    elif a.action == "stop":
        stop()
        print("stopped")
    else:
        print(json.dumps(current(), indent=1))


if __name__ == "__main__":
    sys.exit(main())
